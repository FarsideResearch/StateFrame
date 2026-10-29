"""Host-owned IDEALPATH1 reducer for evidence-driven repair work.

 The reducer is intentionally independent of provider prose. Typed stage and
 localization actions are advertised, dispatched, and resumed by this module;
host receipts are the only source of execution, freshness, and certification
facts. Existing QGATES, REQWITNESS, REPROFIRST, and completion owners remain
reusable adapters, not competing authorities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence


SCHEMA = "stateframe-idealpath/1"
CONTRACT_SCHEMA = "stateframe-idealpath-contract/1"
RECEIPT_SCHEMA = "stateframe-idealpath-check-receipt/1"
TOOL_NAME = "question_gate"
S1_COMPLETION_TOOL_NAME = "requirement_table"
ABSTAIN_ACTION = "abstain"
LOCALIZE_TOOL_NAME = "localize"
CAUSAL_LOCALIZE_TOOL_NAME = "localization_frontier"
TESTWRITER_TOOL_NAME = "typed_test_writer"
S2_FORCE_ALTERNATIVE_ACTION = "localization_alternative"
S3_FORCE_SCHEMA = "stateframe-s3force/1"
FRAMEBUILD_RECEIPT_SCHEMA = "frame-lab/framebuild-owner-receipt/1"
DEFAULT_STAGE_EXIT_BUDGET = 3
STAGES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
STAGE_EXIT_ACTIONS = {
    "S1": frozenset({S1_COMPLETION_TOOL_NAME}),
    # Non-forced S2 projections must expose a localization exit.  The generic
    # question gate remains an admitted compatibility exit and the deterministic
    # budget fallback below continues to use it.
    "S2": frozenset({TOOL_NAME, LOCALIZE_TOOL_NAME, CAUSAL_LOCALIZE_TOOL_NAME}),
    "S3": frozenset({TOOL_NAME}),
    "S4": frozenset({TOOL_NAME}),
    "S5": frozenset({TOOL_NAME}),
    "S6": frozenset({TOOL_NAME}),
    "S7": frozenset({TOOL_NAME}),
}
STAGE_BUDGET_EXIT_ACTIONS = {
    "S1": S1_COMPLETION_TOOL_NAME,
    "S2": TOOL_NAME,
    "S3": TOOL_NAME,
    "S4": TOOL_NAME,
    "S5": TOOL_NAME,
    "S6": TOOL_NAME,
    "S7": TOOL_NAME,
}
OUTCOMES = (
    "PASS",
    "BEHAVIOR_FAIL",
    "INVALID_CHECK",
    "ENVIRONMENT",
    "RESOURCE_LIMIT",
    "NOT_RUN",
    "UNKNOWN",
)
FAILURE_PHASES = (
    "NONE",
    "COLLECTION",
    "IMPORT",
    "EXECUTION",
    "ASSERTION",
    "OBSERVATION",
    "TIMEOUT",
    "UNKNOWN",
)
CERTIFICATION_STATES = ("UNDECIDED", "ADMITTED", "REJECTED", "CERTIFIED")
BACKTRACK_TARGETS = {
    "S2": frozenset({"S1"}),
    "S3": frozenset({"S1", "S2"}),
    "S4": frozenset({"S2", "S3"}),
    "S5": frozenset({"S4"}),
    "S6": frozenset({"S3", "S4", "S5"}),
    "S7": frozenset({"S4", "S5", "S6"}),
}
SOURCE_READ_ACTIONS = frozenset({
    "inspect", "read", "grep", "list", "ls", "list_files", "where_defined", "who_calls",
    "references", "recall_evidence", "recall_generated_artifact", "read_ticket",
})
DIAGNOSTIC_ACTIONS = frozenset({
    "probe", "run_tests", "test", "typed_check_spec", TESTWRITER_TOOL_NAME,
})
PRODUCT_WRITE_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
STAGE_ACTIONS = {
    "S1": frozenset({
        *SOURCE_READ_ACTIONS, "probe", TOOL_NAME, S1_COMPLETION_TOOL_NAME, LOCALIZE_TOOL_NAME,
    }),
    "S2": frozenset({
        *SOURCE_READ_ACTIONS, "probe", TOOL_NAME, LOCALIZE_TOOL_NAME,
        CAUSAL_LOCALIZE_TOOL_NAME,
    }),
    "S3": frozenset({*SOURCE_READ_ACTIONS, *DIAGNOSTIC_ACTIONS, TOOL_NAME}),
    "S4": frozenset({*SOURCE_READ_ACTIONS, "probe", TOOL_NAME}),
    "S5": frozenset({*SOURCE_READ_ACTIONS, *DIAGNOSTIC_ACTIONS,
                     *PRODUCT_WRITE_ACTIONS, TOOL_NAME}),
    "S6": frozenset({*SOURCE_READ_ACTIONS, "probe", "run_tests", "done", TOOL_NAME}),
    "S7": frozenset({*SOURCE_READ_ACTIONS, "done", TOOL_NAME}),
}
REASON_CODES = frozenset({
    "S1_ADMITTED", "S2_ADMITTED", "S3_ADMITTED", "S4_ADMITTED",
    "S5_ADMITTED", "S6_ADMITTED", "S7_ADMITTED", "INVALID_ANSWER",
    "STALE_EPOCH", "STALE_BASE", "STALE_REPLAN", "UNKNOWN_ROW", "MISSING_RECEIPT",
    "ENVIRONMENT_NOT_REPRODUCED", "BACKTRACK_ADMITTED", "BACKTRACK_INVALID",
    "CERTIFICATION_REQUIRED", "HOST_BLOCKED", "CONTRACT_REJECTED",
    "STAGE_CAPABILITY_REJECTED", "S2FORCE_ADMITTED", "S2FORCE_OVERRIDE",
    "S3FORCE_ADMITTED", "STAGE_EXIT_EXHAUSTED", "INSPECT_DISPATCHED",
    "INSPECT_HANDLER_FAILED", "S1_SEMANTIC_REVIEW_UNKNOWN",
    "S1_SEMANTIC_REVIEW_CONTRADICTED", "HOST_CAPABILITY_GAP",
})
_FORBIDDEN = re.compile(
    r"(?:hidden|gold|backup|private|reference\s+fix|reference\s+patch|solution\s+file)",
    re.IGNORECASE,
)
_IMPORT_FAILURE = re.compile(
    r"(?:ImportError|ModuleNotFoundError|No module named|cannot import name|"
    r"collection error|test collection failed)",
    re.IGNORECASE,
)
_RESOURCE_FAILURE = re.compile(r"(?:timeout|timed out|out of memory|resource limit)", re.IGNORECASE)


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text(value: object, *, limit: int = 2000) -> str:
    if not isinstance(value, str):
        return ""
    result = value.replace("\x00", " ").strip()
    if len(result) > limit or _FORBIDDEN.search(result):
        return ""
    return result


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _nonempty_text(value: object) -> bool:
    return bool(_text(value))


def _refs(value: object, *, allow_empty: bool = False) -> bool:
    values = _items(value)
    return (allow_empty and not values) or bool(values) and all(_nonempty_text(item) for item in values)


def _field_schema(field: str) -> dict[str, object]:
    if field.endswith("_refs") or field in {
        "row_ids", "entry_refs", "decision_refs", "outcome_refs", "product_refs",
        "fixtures", "monitors", "regression_test_refs", "regression_refs", "contract_changes",
        "predicted_checks", "quality_check_refs", "verification_refs",
        "unresolved_row_ids", "residual_risks", "limitations", "preserved_invariants",
        "risks", "scope_exceptions",
    }:
        return {"type": "array", "items": {"type": ["string", "object"]}, "maxItems": 64}
    if field in {
        "rows", "edges", "unresolved_edges", "causal_links", "hunks",
        "risk_dispositions", "changes",
    }:
        return {"type": "array", "items": {"type": "object"}, "maxItems": 64}
    if field in {
        "scenario", "oracle", "entry_hint", "assumptions", "alternative",
        "proposed_change", "wrong_repair_control", "sanity_control",
        "cause_proposal", "edit_proposal", "verification_manifest", "finish_proposal",
        "focus", "ticket_focus", "proposal",
    }:
        return {"type": "object"}
    return {"type": "string", "maxLength": 2000}


_ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    "S1": ("rows", "entry_hint", "assumptions"),
    "S2": (
        "row_ids", "entry_refs", "edges", "decision_refs", "outcome_refs",
        "unresolved_edges", "observation_boundary",
    ),
    "S3": (
        "row_ids", "scenario", "oracle", "product_refs", "fixtures",
        "check_artifact_ref", "monitors", "sanity_control",
        "failure_signature", "plausible_wrong_behavior",
    ),
    "S4": (
        "row_ids", "failure_ref", "causal_links", "alternative",
        "proposed_change", "preserved_invariants", "risks",
    ),
    "S5": (
        "cause_receipt", "base_revision", "patch_ref", "hunks",
        "regression_test_refs", "contract_changes", "predicted_checks",
        "scope_exceptions",
    ),
    "S6": (
        "candidate_revision", "rows", "regression_refs", "wrong_repair_control",
        "quality_check_refs", "risk_dispositions",
    ),
    "S7": (
        "outcome", "artifact_ref", "changes", "verification_refs",
        "unresolved_row_ids", "residual_risks", "limitations",
    ),
}

_OPTIONAL_ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    "S1": ("focus", "ticket_focus", "proposal"),
    "S4": ("cause_proposal",),
    "S5": ("edit_proposal",),
    "S6": ("verification_manifest",),
    "S7": ("finish_proposal",),
}


def _answer_schema(stage: str) -> dict[str, object]:
    fields = _ANSWER_FIELDS[stage]
    optional = _OPTIONAL_ANSWER_FIELDS.get(stage, ())
    properties = {field: _field_schema(field) for field in (*fields, *optional)}
    if stage == "S7":
        properties["outcome"] = {
            "type": "string",
            "enum": ["VERIFIED_FIXED", "NOT_REPRODUCED", "INCOMPLETE"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


def tool_schema() -> dict[str, object]:
    """Return the single discriminated gate schema used by menu and dispatch."""
    variants = []
    for stage in STAGES:
        variants.append({
            "type": "object",
            "properties": {
                "stage": {"const": stage},
                "state_epoch": {"type": "integer", "minimum": 1},
                "ticket_hash": {"type": "string", "minLength": 1},
                "base_revision": {"type": "string", "minLength": 1},
                "evidence_refs": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 64,
                },
                "answer": _answer_schema(stage),
            },
            "required": [
                "stage", "state_epoch", "ticket_hash", "base_revision",
                "evidence_refs", "answer",
            ],
            "additionalProperties": False,
        })
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": (
                "Submit one typed IDEALPATH1 state answer; prose cannot settle a state."
            ),
            "parameters": {"oneOf": variants},
        },
    }


def stage_tool_schema(stage: str) -> dict[str, object]:
    """Return the current-stage variant, keeping the live menu discriminated."""
    if stage not in STAGES:
        raise ValueError(f"unknown IDEALPATH stage: {stage}")
    variant = {
        "type": "object",
        "properties": {
            "stage": {"const": stage},
            "state_epoch": {"type": "integer", "minimum": 1},
            "ticket_hash": {"type": "string", "minLength": 1},
            "base_revision": {"type": "string", "minLength": 1},
            "evidence_refs": {
                "type": "array", "items": {"type": "string"}, "maxItems": 64,
            },
            "answer": _answer_schema(stage),
        },
        "required": [
            "stage", "state_epoch", "ticket_hash", "base_revision",
            "evidence_refs", "answer",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": f"Submit the typed IDEALPATH1 {stage} state answer.",
            "parameters": variant,
        },
    }


def requirement_table_tool_schema(host_draft: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the explicit typed S1 exit contract.

    ``question_gate`` remains available for compatibility, but S1's live menu
    must name the operation that registers the complete ticket-backed table.
    """
    answer_properties = _answer_schema("S1")["properties"]
    assert isinstance(answer_properties, Mapping)
    properties = {
        "stage": {"const": "S1"},
        "state_epoch": {"type": "integer", "minimum": 1},
        "ticket_hash": {"type": "string", "minLength": 1},
        "base_revision": {"type": "string", "minLength": 1},
        "evidence_refs": {
            "type": "array", "items": {"type": "string"}, "maxItems": 64,
        },
        "rows": copy.deepcopy(answer_properties["rows"]),
        "entry_hint": copy.deepcopy(answer_properties["entry_hint"]),
        "assumptions": copy.deepcopy(answer_properties["assumptions"]),
        "proposal": {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 64},
            },
            "required": ["rows"],
            "additionalProperties": False,
        },
    }
    description = (
        "Register the complete typed S1 requirement table. Every row must "
        "cite exact public ticket span references before S1 can advance."
    )
    if isinstance(host_draft, Mapping):
        draft_view = {
            "spine_schema": host_draft.get("spine_schema"),
            "rows": copy.deepcopy(host_draft.get("rows", ())),
            "entry_hint": copy.deepcopy(host_draft.get("entry_hint", {})),
            "assumptions": copy.deepcopy(host_draft.get("assumptions", ())),
            "provenance": host_draft.get("provenance"),
        }
        rendered = json.dumps(draft_view, ensure_ascii=False, sort_keys=True, default=str)
        description += " Host draft to confirm or edit: " + rendered[:7_000]
    return {
        "type": "function",
        "function": {
            "name": S1_COMPLETION_TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    "stage", "state_epoch", "ticket_hash", "base_revision",
                    "evidence_refs", "rows", "entry_hint", "assumptions",
                ],
                "additionalProperties": False,
            },
        },
    }


class StageExitDeadlock(RuntimeError):
    """Raised when a projected stage menu cannot satisfy its exit predicate."""


def stage_exit_liveness(
        stage: str,
        offered_actions: Sequence[object],
        admitted_actions: Sequence[object] | None = None,
) -> dict[str, object]:
    """Check the GENERAL stage-exit invariant after menu projection."""
    if stage not in STAGES:
        raise ValueError(f"unknown IDEALPATH stage: {stage}")
    offered = {str(action) for action in offered_actions}
    admitted = offered if admitted_actions is None else {str(action) for action in admitted_actions}
    required = STAGE_EXIT_ACTIONS[stage]
    satisfying = sorted(required & offered & admitted)
    return {
        "schema": "stateframe-stage-exit-liveness/1",
        "invariant": "GENERAL_STAGE_EXIT_ACTION_LIVENESS",
        "stage": stage,
        "required_action_ids": sorted(required),
        "offered_action_ids": sorted(offered),
        "admitted_action_ids": sorted(admitted),
        "satisfying_action_ids": satisfying,
        "live": bool(satisfying),
    }


def assert_stage_exit_liveness(
        stage: str,
        offered_actions: Sequence[object],
        admitted_actions: Sequence[object] | None = None,
) -> dict[str, object]:
    """Fail closed instead of returning a navigation-only stage menu."""
    report = stage_exit_liveness(stage, offered_actions, admitted_actions)
    if not report["live"]:
        raise StageExitDeadlock(
            f"{stage} projected menu has no admitted exit action: "
            f"{report['required_action_ids']}"
        )
    return report


def assert_all_stage_exit_liveness(
        projected_menus: Mapping[str, Sequence[object]],
) -> dict[str, dict[str, object]]:
    """Screen every S1-S7 menu and fail if any stage is unexitable."""
    reports: dict[str, dict[str, object]] = {}
    for stage in STAGES:
        actions = projected_menus.get(stage, ())
        reports[stage] = assert_stage_exit_liveness(stage, actions, actions)
    return reports


def backtrack_tool_schema(targets: Sequence[str]) -> dict[str, object] | None:
    """Return the typed backtrack action for one reducer menu.

    ``None`` when no admitted target exists: ``enum: []`` is an HTTP 500 on
    llama.cpp (audit 2026-10-05 #1), so the tool is dropped instead.
    """
    allowed = sorted(str(target) for target in targets if str(target) in STAGES)
    if not allowed:
        return None
    return {
        "type": "function",
        "function": {
            "name": "backtrack",
            "description": "Move the host-owned IDEALPATH state to an admitted earlier stage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "enum": allowed},
                    "reason_code": {"type": "string", "enum": sorted(REASON_CODES)},
                },
                "required": ["target", "reason_code"],
                "additionalProperties": False,
            },
        },
    }


def localize_tool_schema() -> dict[str, object]:
    """Return the targetless S2 search/bind action.

    ``path`` remains supported for the established NOTARGET edit flow, while
    omitting it performs host-owned retrieval and returns a typed suspect page.
    """
    return {
        "type": "function",
        "function": {
            "name": LOCALIZE_TOOL_NAME,
            "description": (
                "Bind an existing product file discovered in the allowed workspace scope. "
                "Do not choose a file alphabetically; provide the path and evidence refs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 400},
                    "symbol": {"type": "string", "maxLength": 200},
                    "query": {"type": "string", "maxLength": 2400},
                    "channel": {
                        "type": "string",
                        "enum": ["literal", "lexical", "entrypoint", "state", "public_test"],
                    },
                    "page": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 20},
                    "cursor": {"type": "string", "maxLength": 32},
                    "candidate_id": {"type": "string", "maxLength": 400},
                    "evidence_refs": {
                        "type": "array", "items": {"type": "string"}, "maxItems": 32,
                    },
                    "reason": {"type": "string", "maxLength": 500},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def _classify_outcome(
        status: object,
        *,
        stdout: object = "",
        stderr: object = "",
        reached_observation: object = None,
        failure_phase: object = None,
        outcome: object = None,
) -> tuple[str, str, bool | None]:
    output = "\n".join(value for value in (stdout, stderr) if isinstance(value, str))
    if _IMPORT_FAILURE.search(output) or str(failure_phase or "").upper() in {"IMPORT", "COLLECTION"}:
        return "ENVIRONMENT", "IMPORT" if "IMPORT" in output.upper() else "COLLECTION", False
    explicit = str(outcome or "").upper()
    if explicit in OUTCOMES:
        selected = explicit
    else:
        raw = str(status or "").upper()
        selected = {
            "PASS": "PASS", "PASSED": "PASS", "GREEN": "PASS", "OK": "PASS",
            "FAIL": "BEHAVIOR_FAIL", "FAILED": "BEHAVIOR_FAIL", "RED": "BEHAVIOR_FAIL",
            "INVALID_CHECK": "INVALID_CHECK", "NOT_RUN": "NOT_RUN", "UNKNOWN": "UNKNOWN",
            "ENVIRONMENT": "ENVIRONMENT", "ENV_ERROR": "ENVIRONMENT",
        }.get(raw, "UNKNOWN")
        if _RESOURCE_FAILURE.search(output):
            selected = "RESOURCE_LIMIT"
    phase = str(failure_phase or "").upper()
    if phase not in FAILURE_PHASES:
        phase = {
            "BEHAVIOR_FAIL": "ASSERTION",
            "PASS": "NONE",
            "INVALID_CHECK": "COLLECTION",
            "ENVIRONMENT": "EXECUTION",
            "RESOURCE_LIMIT": "TIMEOUT",
        }.get(selected, "UNKNOWN")
    reached = reached_observation if isinstance(reached_observation, bool) else None
    if reached is None and selected == "BEHAVIOR_FAIL":
        reached = phase in {"ASSERTION", "OBSERVATION"}
    if reached is None and selected == "PASS":
        reached = True
    if selected == "ENVIRONMENT":
        reached = False
    return selected, phase, reached


def _typed_receipt(
        *,
        receipt_id: str,
        requirement_id: str | None,
        status: object,
        product_revision: str,
        current_revision: str,
        check_hash: str,
        stdout: object,
        stderr: object,
        reached_observation: object,
        failure_phase: object,
        outcome: object,
        ticket_relevant: bool,
) -> dict[str, object]:
    typed_outcome, phase, reached = _classify_outcome(
        status,
        stdout=stdout,
        stderr=stderr,
        reached_observation=reached_observation,
        failure_phase=failure_phase,
        outcome=outcome,
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "requirement_id": requirement_id,
        "result": {
            "status": str(status or "UNKNOWN").upper(),
            "outcome": typed_outcome,
            "reached_observation": reached,
            "failure_phase": phase,
            "ticket_relevant": bool(ticket_relevant),
            "failure_signature": _text(stderr or stdout, limit=1200),
        },
        "freshness": {
            "product_revision": product_revision,
            "current_revision": current_revision,
            "check_hash": check_hash,
            "is_current": product_revision == current_revision,
            "integrity": True,
        },
        "certification": {
            "status": "UNDECIDED",
            "admitted": False,
            "basis": None,
        },
    }


class IdealPathOwner:
    """One host-owned reducer for S1 through S7 and typed backtracking."""

    def __init__(
            self,
            *,
            ticket_hash: str,
            base_revision: str,
            ticket_spans: Sequence[str] = (),
            owner_id: str = "host-idealpath",
            max_actions: int = 64,
            host_draft: Mapping[str, object] | None = None,
            semantic_reviewer: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
            stage_exit_budget: int = DEFAULT_STAGE_EXIT_BUDGET,
            inspect_handler: Callable[[Mapping[str, object]], object] | None = None,
    ) -> None:
        if not _nonempty_text(ticket_hash) or not _nonempty_text(base_revision):
            raise ValueError("ticket_hash and base_revision are host-required")
        self.ticket_hash = ticket_hash
        self.base_revision = base_revision
        self.ticket_spans = frozenset(
            str(item) for item in ticket_spans if _nonempty_text(item)
        )
        self.owner_id = owner_id
        self.max_actions = max_actions
        if type(stage_exit_budget) is not int or stage_exit_budget < 1:
            raise ValueError("stage_exit_budget must be a positive integer")
        self.stage_exit_budget = stage_exit_budget
        from kernel_l109_accounting import LateStageAccounting
        self.late_stage_accounting = LateStageAccounting(
            capacity=stage_exit_budget, pending_limit=stage_exit_budget,
        )
        self.stage = "S1"
        self.epoch = 1
        self.rows: dict[str, dict[str, object]] = {}
        self.receipts: dict[str, dict[str, object]] = {}
        self.localization_receipts: dict[str, dict[str, object]] = {}
        self.transitions: list[dict[str, object]] = []
        self.answers: dict[str, dict[str, object]] = {}
        self.ticket_focus: object = None
        self.admitted_failures: dict[str, str] = {}
        # S3 freezes the host-generated discriminator and carries its digest
        # into S6; these fields are included in every IDEALPATH snapshot.
        self.frozen_repros: dict[str, dict[str, object]] = {}
        self.typed_repro_attempts: dict[str, int] = {}
        # The requirements ledger owns these references.  IDEALPATH keeps only
        # a read-only identity projection and stage-consumption receipts.
        self.requirement_table: dict[str, object] | None = None
        self.requirement_table_receipt: dict[str, object] | None = None
        self.spine_stage_records: dict[str, set[tuple[str, int]]] = {}
        self._spine_row_refs: set[tuple[str, int]] = set()
        self.requirement_ledger: dict[str, object] | None = None
        self.semantic_review_receipt: dict[str, object] | None = None
        self.semantic_reviewer = semantic_reviewer
        self.admitted_parent_passes: dict[str, str] = {}
        self.authored_witnesses: dict[str, str] = {}
        self.s3_force_attempts: dict[str, dict[str, object]] = {}
        self.s3_force_row_cursor = 0
        self.s3_force_settlement: dict[str, object] | None = None
        self._settled_stage_requests: set[str] = set()
        self._productive_receipts: set[str] = set()
        self.pending_stage_capacity = 0
        self.verification: dict[str, object] | None = None
        # S47 is an additive host-contract seam.  Legacy answers remain
        # readable, while typed answers are admitted only after the host
        # registry and the corresponding S47 predicate accept them.
        self.s47_registry: Mapping[str, object] | None = None
        self.s47_cause_admission: dict[str, object] | None = None
        self.s47_edit_permit: dict[str, object] | None = None
        self.s47_verification_context: Mapping[str, object] = {}
        self.s47_verification_decision: dict[str, object] | None = None
        self.action_count = 0
        self.stage_turns: dict[str, int] = {stage: 0 for stage in STAGES}
        self._last_stage_turn_marker: tuple[str, object] | None = None
        self._capability_snapshot = None
        self._stale_batch_gate = None
        self.host_draft = copy.deepcopy(dict(host_draft)) if isinstance(host_draft, Mapping) else None
        if inspect_handler is not None and not callable(inspect_handler):
            raise ValueError("inspect_handler must be callable")
        if semantic_reviewer is not None and not callable(semantic_reviewer):
            raise ValueError("semantic_reviewer must be callable")
        self.inspect_handler = inspect_handler
        self.navigation_evidence: list[dict[str, object]] = []
        self.stage_evidence: dict[str, list[dict[str, object]]] = {}
        self.abstentions: list[dict[str, object]] = []
        self.s2_force_frontier: dict[str, object] | None = None
        self.s2_force_target: dict[str, str] = {"path": "", "symbol": ""}
        self.s2_force_receipts: list[dict[str, object]] = []
        self.s3_force_receipts: list[dict[str, object]] = []
        self._refusal_tool = ""
        self._refusal_streak = 0
        self._s2_force_menu_notice: dict[str, object] | None = None
        self._s3_force_menu_notice: dict[str, object] | None = None
        self._framebuild_receipt: dict[str, object] = {
            "schema": FRAMEBUILD_RECEIPT_SCHEMA,
            "consumer": "idealpath",
            "owner_id": self.owner_id,
            "builder_version": None,
            "frame_sha256": None,
            "request_sha256": None,
            "status": "UNBOUND",
        }

    def bind_framebuild_frame(self, frame: object) -> dict[str, object]:
        """Bind one canonical public frame to this reducer owner.

        The import is lazy so frozen IDEALPATH releases remain usable when the
        optional request qualification package is not bundled into a host process.
        """
        builder_version = getattr(frame, "builder_version", None)
        frame_sha256 = getattr(frame, "frame_sha256", None)
        if not isinstance(builder_version, str) or not builder_version:
            raise ValueError("FRAMEBUILD frame is missing builder_version")
        if not isinstance(frame_sha256, str) or not frame_sha256:
            raise ValueError("FRAMEBUILD frame is missing frame_sha256")
        request_sha256 = getattr(frame, "request_sha256", None)
        self._framebuild_receipt = {
            "schema": FRAMEBUILD_RECEIPT_SCHEMA,
            "consumer": "idealpath",
            "owner_id": self.owner_id,
            "builder_version": builder_version,
            "frame_sha256": frame_sha256,
            "request_sha256": request_sha256 if isinstance(request_sha256, str) else None,
            "status": "BOUND",
        }
        return self.framebuild_receipt()

    def build_framebuild_frame(
            self,
            task: object,
            *,
            public_repo: object | None = None,
            host_execution: object | None = None,
    ) -> object:
        """Build and bind IDEALPATH's public-evidence frame through FRAMEBUILD."""
        from kernel_optional_engine_module import optional_module
        framebuild = optional_module("STATEFRAME_FRAMEBUILD_MODULE")
        if framebuild is None:
            raise ImportError("frame builder extension is not configured")
        build_consumer_frame = framebuild.build_consumer_frame

        frame = build_consumer_frame(
            "idealpath",
            task,
            public_repo=public_repo,
            host_execution=host_execution,
        )
        self.bind_framebuild_frame(frame)
        return frame

    def framebuild_receipt(self) -> dict[str, object]:
        """Return the owner receipt carried by menus and snapshots."""
        receipt = dict(self._framebuild_receipt)
        receipt.update({"stage": self.stage, "state_epoch": self.epoch})
        return receipt

    def set_s2_force_context(
            self, *, frontier: Mapping[str, object] | None = None,
            target_path: str = "", target_symbol: str = "",
    ) -> None:
        """Refresh host-only S2 inputs used if the stage budget is exhausted."""
        self.s2_force_frontier = (
            copy.deepcopy(dict(frontier)) if isinstance(frontier, Mapping) else None
        )
        self.s2_force_target = {
            "path": _text(target_path, limit=500),
            "symbol": _text(target_symbol, limit=300),
        }

    def note_refusal(self, action: str, *, reason_code: str,
                     admitted_actions: Sequence[object]) -> dict[str, object] | None:
        """Track one same-tool refusal and prepare a bounded menu notice."""
        from kernel_s2_force import DEFAULT_REJECT_STREAK, enabled as s2_enabled, reject_loop_notice
        from kernel_s3_force import enabled as s3_enabled

        if self.stage == "S2" and not s2_enabled():
            return None
        if self.stage == "S3" and not s3_enabled():
            return None
        if self.stage not in {"S2", "S3"}:
            return None
        name = str(action or "").strip()
        self._refusal_streak = self._refusal_streak + 1 if name == self._refusal_tool else 1
        self._refusal_tool = name
        if self._refusal_streak < DEFAULT_REJECT_STREAK:
            return None
        if self.stage == "S2":
            self._s2_force_menu_notice = reject_loop_notice(
                tool=name,
                streak=self._refusal_streak,
                admitted_actions=admitted_actions,
            )
            self._s2_force_menu_notice["reason_code"] = str(reason_code)
            return copy.deepcopy(self._s2_force_menu_notice)
        self._s3_force_menu_notice = {
            "schema": S3_FORCE_SCHEMA,
            "code": "S3_FORCE_RERENDER",
            "tool": name,
            "streak": self._refusal_streak,
            "admitted_actions": [str(item) for item in admitted_actions],
            "message": f"S3FORCE:{name} refused x{self._refusal_streak}; use typed_test_writer or abstain",
            "reason_code": str(reason_code),
        }
        return copy.deepcopy(self._s3_force_menu_notice)

    def clear_refusal_streak(self) -> None:
        self._refusal_tool = ""
        self._refusal_streak = 0

    def s2_force_menu_notice(self) -> dict[str, object] | None:
        notice = (
            self._s3_force_menu_notice
            if self.stage == "S3"
            else self._s2_force_menu_notice
        )
        return copy.deepcopy(notice)

    def s3_force_menu_notice(self) -> dict[str, object] | None:
        return copy.deepcopy(self._s3_force_menu_notice)

    def bind_s47_registry(self, registry: Mapping[str, object]) -> dict[str, object]:
        """Bind host-issued S47 identities without copying model authority."""
        if not isinstance(registry, Mapping):
            raise ValueError("S47 registry must be a mapping")
        if registry.get("state_epoch") != self.epoch or registry.get("base_ref") != self.base_revision:
            raise ValueError("S47 registry is stale for this IDEALPATH epoch/base")
        self.s47_registry = copy.deepcopy(dict(registry))
        return copy.deepcopy(dict(registry))

    def admit_s47_cause(self, proposal: Mapping[str, object], *,
                        review: Mapping[str, object] | None = None) -> dict[str, object]:
        """Run the typed S4 host admission when a proposal is supplied."""
        if self.s47_registry is None:
            return {"valid": False, "status": "BLOCKED", "missing": ["s47_registry"]}
        from kernel_s47_causal_fix import admit_cause
        host_review = review
        if host_review is None:
            candidate_review = self.s47_registry.get("cause_review")
            host_review = candidate_review if isinstance(candidate_review, Mapping) else None
        result = admit_cause(proposal, self.s47_registry, review=host_review)
        self.s47_cause_admission = copy.deepcopy(dict(result))
        return copy.deepcopy(dict(result))

    def bind_s47_edit_permit(self, permit: Mapping[str, object]) -> dict[str, object]:
        """Retain only a host-issued S5 authorization identity for later projections."""
        from kernel_s47_causal_fix import PERMIT_SCHEMA
        if not isinstance(permit, Mapping) or permit.get("schema") != PERMIT_SCHEMA:
            raise ValueError("invalid S47 edit authorization")
        self.s47_edit_permit = copy.deepcopy(dict(permit))
        return copy.deepcopy(dict(permit))

    def bind_s47_verification_context(self, context: Mapping[str, object]) -> None:
        """Bind host receipts used by the typed S6 conjunction."""
        if not isinstance(context, Mapping):
            raise ValueError("S47 verification context must be a mapping")
        self.s47_verification_context = copy.deepcopy(dict(context))

    def verify_s47_manifest(self, manifest: Mapping[str, object],
                            context: Mapping[str, object]) -> dict[str, object]:
        """Evaluate S6's host conjunction and retain its decision only."""
        from kernel_s47_causal_fix import verify_manifest
        if not isinstance(context, Mapping):
            raise ValueError("S47 verification context must be a mapping")
        self.s47_verification_context = copy.deepcopy(dict(context))
        result = verify_manifest(manifest, self.s47_verification_context)
        self.s47_verification_decision = copy.deepcopy(dict(result))
        return copy.deepcopy(dict(result))

    def settle_s47_finish(self, finish: Mapping[str, object], *,
                          artifact: Mapping[str, object] | None = None) -> dict[str, object]:
        """Settle S7 through the shared host terminal contract."""
        from kernel_s47_causal_fix import settle_terminal
        if self.s47_verification_decision is None:
            decision = {"status": "INCOMPLETE", "blockers": ["NO_S6_DECISION"]}
        else:
            decision = self.s47_verification_decision
        state = getattr(self, "s47_terminal_state", None)
        if not isinstance(state, dict):
            state = {}
            self.s47_terminal_state = state
        result = settle_terminal(finish, decision, artifact=artifact, state=state)
        self.s47_finish_proposal = copy.deepcopy(dict(finish))
        self.s47_terminal_disposition = copy.deepcopy(dict(result))
        return copy.deepcopy(dict(result))

    def bind_requirement_table(self, projection: Mapping[str, object]) -> dict[str, object]:
        """Bind the reducer to host-issued ledger row references.

        This is a projection seam: it never accepts row semantics or IDs from
        the model and never creates a second requirement collection.
        """
        if projection.get("schema") != "requirement-spine-table/1":
            raise ValueError("invalid requirement spine projection")
        required = {"schema", "ledger_id", "table_revision", "table_hash", "active_rows"}
        if set(projection) != required:
            raise ValueError("requirement spine projection fields are closed")
        ledger_id = projection.get("ledger_id")
        table_hash = projection.get("table_hash")
        revision = projection.get("table_revision")
        if not _nonempty_text(ledger_id) or not _nonempty_text(table_hash):
            raise ValueError("requirement spine projection identity is required")
        if type(revision) is not int or revision < 1:
            raise ValueError("requirement spine projection revision is invalid")
        refs: set[tuple[str, int]] = set()
        for item in _items(projection.get("active_rows")):
            if not isinstance(item, Mapping):
                raise ValueError("requirement spine active rows must be objects")
            if item.get("ledger_id") != ledger_id or not _nonempty_text(item.get("id")):
                raise ValueError("requirement spine row identity is invalid")
            version = item.get("version")
            if type(version) is not int or version < 1:
                raise ValueError("requirement spine row version is invalid")
            refs.add((str(item["id"]), version))
        if not refs:
            raise ValueError("requirement spine projection needs active rows")
        self.requirement_table = copy.deepcopy(dict(projection))
        self.spine_stage_records = {}
        self._spine_row_refs = refs
        return copy.deepcopy(dict(projection))

    def consume_requirement_stage(self, record: Mapping[str, object]) -> dict[str, object]:
        """Record stage row consumption against the bound ledger identity."""
        if self.requirement_table is None:
            return self._reply(False, code="HOST_BLOCKED", missing=("requirement_table",))
        if record.get("schema") != "requirement-spine-stage-record/1":
            return self._reply(False, code="INVALID_ANSWER", missing=("stage_record_schema",))
        stage = str(record.get("stage", ""))
        if stage not in STAGES[1:]:
            return self._reply(False, code="INVALID_ANSWER", missing=("stage_record_stage",))
        if (
            record.get("ledger_id") != self.requirement_table.get("ledger_id")
            or record.get("table_revision") != self.requirement_table.get("table_revision")
            or record.get("table_hash") != self.requirement_table.get("table_hash")
        ):
            return self._reply(False, code="STALE_BASE", missing=("requirement_table_identity",))
        refs: set[tuple[str, int]] = set()
        for item in _items(record.get("row_refs")):
            if not isinstance(item, Mapping) or item.get("ledger_id") != self.requirement_table.get("ledger_id"):
                return self._reply(False, code="UNKNOWN_ROW", missing=("canonical_row_ref",))
            version = item.get("version")
            if type(version) is not int or version < 1:
                return self._reply(False, code="UNKNOWN_ROW", missing=("canonical_row_version",))
            refs.add((str(item.get("id", "")), version))
        if not refs or not refs.issubset(getattr(self, "_spine_row_refs", set())):
            return self._reply(False, code="UNKNOWN_ROW", missing=("canonical_row_ref",))
        previous = self.spine_stage_records.setdefault(stage, set())
        if previous & refs:
            return self._reply(False, code="INVALID_ANSWER", missing=("duplicate_stage_row",))
        previous.update(refs)
        return self._reply(True, code=f"{stage}_ROW_REFS_CONSUMED", status="ADMITTED")

    def record_localization(self, result: Mapping[str, object]) -> dict[str, object]:
        """Persist one host search receipt without admitting S2 by shape alone."""
        if str(result.get("schema", "")) != "stateframe-localize/1":
            return {"accepted": False, "code": "LOCALIZATION_SCHEMA_REJECTED"}
        receipt = result.get("receipt")
        if not isinstance(receipt, Mapping) or not _nonempty_text(receipt.get("receipt_id")):
            return {"accepted": False, "code": "LOCALIZATION_RECEIPT_REQUIRED"}
        receipt_id = str(receipt["receipt_id"])
        self.localization_receipts[receipt_id] = dict(copy.deepcopy(dict(receipt)))
        return {
            "accepted": True,
            "code": "LOCALIZATION_RECEIPT_RECORDED",
            "receipt_id": receipt_id,
            "snapshot_id": receipt.get("snapshot_id"),
            "source_receipt_ids": list(receipt.get("source_receipt_ids", ()))[:64],
        }

    @property
    def terminal(self) -> bool:
        return self.stage == "S7"

    def settle_stage_action(
            self, request_id: object, *, action: str = "request",
            receipt_ids: Sequence[object] = (), pending_checks: int = 0,
            refusal: bool = False,
    ) -> dict[str, object]:
        """Settle one late-stage request/action through the shared meter."""
        accounting = self.late_stage_accounting.settle(
            f"{self.stage}:{request_id}", action=action,
            receipt_ids=receipt_ids, pending_checks=pending_checks, refusal=refusal,
        )
        self.pending_stage_capacity = int(accounting.get("pending", 0))
        return accounting

    def observe_stage_turn(
            self, marker: object = None, *, action: str = "request",
            receipt_ids: Sequence[object] = (), pending_checks: int = 0,
            refusal: bool = False,
    ) -> dict[str, object]:
        """Count settled requests once and expose bounded pending capacity."""
        marker_value = marker if marker is not None else self.stage_turns[self.stage] + 1
        key = (self.stage, marker_value)
        accounting = self.settle_stage_action(
            marker_value, action=action, receipt_ids=receipt_ids,
            pending_checks=pending_checks, refusal=refusal,
        )
        if key != self._last_stage_turn_marker:
            self.stage_turns[self.stage] = self.stage_turns.get(self.stage, 0) + 1
            self._last_stage_turn_marker = key
        budget = self.stage_budget()
        budget["accounting"] = accounting
        budget["pending_capacity"] = self.pending_stage_capacity
        return budget

    def capability_snapshot(
            self,
            offered_actions: Sequence[object],
            *,
            request_id: object = None,
            budget_generation: object = None,
            product_revision: object = None,
    ) -> dict[str, object]:
        """Freeze one final menu identity after all host enrichers have run."""
        from kernel_fivefix_control import build_capability_snapshot

        capabilities = self.capability_set()
        snapshot = build_capability_snapshot(
            request_id=(request_id if request_id is not None else f"stage:{self.stage}:epoch:{self.epoch}"),
            stage=self.stage,
            epoch=self.epoch,
            source_revision=self.base_revision,
            product_revision=(product_revision if product_revision is not None else self.base_revision),
            budget_generation=(budget_generation if budget_generation is not None else f"stage:{self.stage}:turn:{self.stage_turns[self.stage]}"),
            admitted_actions=capabilities.get("allowed_actions", ()),
            offered_actions=offered_actions,
            exit_action_ids=STAGE_EXIT_ACTIONS[self.stage],
            recovery_action_ids=capabilities.get("recovery_action_ids", ()),
            capability_facts=capabilities,
        )
        self._capability_snapshot = snapshot
        return snapshot.to_dict()

    def validate_batch(self, batch: Mapping[str, object]) -> dict[str, object]:
        """Reject stale native work with one typed replan/cancel decision."""
        from kernel_fivefix_control import StaleBatchGate

        snapshot = self._capability_snapshot
        if snapshot is None:
            return {
                "status": "STALE_REPLAN",
                "reason_code": "STALE_REPLAN",
                "replan": True,
                "cancel": True,
                "detail": "no_frozen_capability_snapshot",
            }
        if self._stale_batch_gate is None:
            self._stale_batch_gate = StaleBatchGate()
        decision = self._stale_batch_gate.validate(snapshot, batch)
        return {
            "status": decision.status,
            "batch_id": decision.batch_id,
            "reason_code": "STALE_REPLAN" if decision.replan else decision.status,
            "reason": decision.reason,
            "replan": decision.replan,
            "cancel": decision.cancel,
            "capability_snapshot": snapshot.to_dict(),
        }

    def stage_budget(self) -> dict[str, object]:
        """Expose host-owned stage budget state without model authority."""
        used = int(self.stage_turns.get(self.stage, 0))
        reserved = min(self.stage_exit_budget, max(0, int(self.pending_stage_capacity)))
        effective_limit = self.stage_exit_budget + reserved
        forced = used >= effective_limit
        return {
            "schema": "stateframe-stage-exit-budget/1",
            "stage": self.stage,
            "used": used,
            "limit": effective_limit,
            "base_limit": self.stage_exit_budget,
            "reserved_pending": reserved,
            "remaining": max(0, effective_limit - used),
            "forced": forced,
            "exit_action": STAGE_BUDGET_EXIT_ACTIONS[self.stage],
        }

    def forced_exit_action(self) -> str | None:
        budget = self.stage_budget()
        return str(budget["exit_action"]) if budget["forced"] else None

    def _forced_actions(self) -> tuple[str, ...]:
        exit_action = self.forced_exit_action()
        if exit_action is None:
            return ()
        if self.stage == "S2":
            from kernel_s2_force import ALTERNATIVE_ACTION, enabled
            if enabled():
                return (exit_action, ALTERNATIVE_ACTION, ABSTAIN_ACTION)
        if self.stage == "S3":
            from kernel_s3_force import enabled
            if enabled():
                # Keep question_gate as the stage exit identity for liveness;
                # the runtime turns it/abstain into the host fallback while
                # typed_test_writer remains the explicit model override.
                actions = [exit_action, TESTWRITER_TOOL_NAME, ABSTAIN_ACTION]
                settlement = self.s3_force_settlement
                if isinstance(settlement, Mapping):
                    missing = " ".join(str(item).lower() for item in _items(settlement.get("missing")))
                    reason = str(settlement.get("reason", "")).lower()
                    if any(token in missing or token in reason for token in (
                            "s1", "s2", "semantic", "localiz", "target packet")):
                        actions.append("backtrack")
                return tuple(actions)
        return (exit_action, ABSTAIN_ACTION)

    def _navigation_arguments(self, arguments: Mapping[str, object]) -> dict[str, object]:
        allowed = {"path", "query", "symbol", "line", "target", "function", "scope", "pattern"}
        return {
            key: (value if type(value) is int else _text(value, limit=300))
            for key, value in arguments.items()
            if key in allowed and (type(value) is int or _nonempty_text(value))
        }

    def note_navigation(self, action: str, arguments: Mapping[str, object] | None = None,
                        observation: object = "", *, step: object = None) -> None:
        """Retain bounded S1 source-navigation evidence for the S2 suspect set."""
        if str(action) not in SOURCE_READ_ACTIONS:
            return
        args = self._navigation_arguments(arguments if isinstance(arguments, Mapping) else {})
        text = _text(observation, limit=1800)
        marker = step if type(step) is int else None
        if self.stage != "S1":
            if self.stage == "S2" and text:
                found = False
                for item in reversed(self.stage_evidence.get("S2", ())):
                    if (
                        item.get("action") == str(action)
                        and item.get("arguments") == args
                        and item.get("step") == marker
                    ):
                        item["observation"] = text
                        found = True
                        break
                if not found:
                    evidence_id = "s2-navigation:" + _digest({
                        "action": str(action), "arguments": args, "step": marker,
                    })[7:31]
                    self.stage_evidence.setdefault("S2", []).append({
                        "schema": "stateframe-s2-model-navigation-evidence/1",
                        "evidence_id": evidence_id,
                        "stage": "S2",
                        "step": marker,
                        "action": str(action),
                        "arguments": args,
                        "observation": text,
                        "classification": "S2_MODEL_EVIDENCE",
                    })
                    self.stage_evidence["S2"] = self.stage_evidence["S2"][-64:]
            return
        signature = _digest({"action": str(action), "arguments": args, "step": marker})
        for item in reversed(self.navigation_evidence):
            if item.get("evidence_id") == "s1-navigation:" + signature[7:31]:
                if text:
                    item["observation"] = text
                return
        self.navigation_evidence.append({
            "schema": "stateframe-s1-navigation-evidence/1",
            "evidence_id": "s1-navigation:" + signature[7:31],
            "stage": "S1",
            "step": marker,
            "action": str(action),
            "arguments": args,
            "observation": text,
            "classification": "S2_SUSPECT_OR_EVIDENCE",
        })
        self.navigation_evidence = self.navigation_evidence[-64:]

    def _host_draft_answer(self) -> dict[str, object] | None:
        draft = self.host_draft
        if not isinstance(draft, Mapping):
            return None
        answer = {
            "rows": copy.deepcopy(draft.get("rows", ())),
            "entry_hint": copy.deepcopy(draft.get("entry_hint", {})),
            "assumptions": copy.deepcopy(draft.get("assumptions", ())),
        }
        for key in ("focus", "ticket_focus"):
            if key in draft:
                answer[key] = copy.deepcopy(draft[key])
        return answer

    def _host_ledger(self) -> Mapping[str, object] | None:
        ledger = self.requirement_ledger
        if isinstance(ledger, Mapping):
            return ledger
        if isinstance(self.host_draft, Mapping):
            candidate = self.host_draft.get("requirement_ledger")
            if isinstance(candidate, Mapping):
                return candidate
        return None

    def row_role(self, row_id: object) -> str:
        """Return the role stamped by the host draft/ledger, never the answer."""
        ident = str(row_id)
        row = self.rows.get(ident, {})
        if isinstance(row, Mapping) and str(row.get("role", "")).upper() in {
                "REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
            return str(row["role"]).upper()
        for source in (self.host_draft, self.requirement_ledger):
            raw_rows = source.get("rows", ()) if isinstance(source, Mapping) else ()
            for candidate in _items(raw_rows):
                if isinstance(candidate, Mapping) and str(candidate.get("row_id")) == ident:
                    role = str(candidate.get("role", "")).upper()
                    if role in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
                        return role
            requirements = source.get("requirements", ()) if isinstance(source, Mapping) else ()
            for candidate in _items(requirements):
                if isinstance(candidate, Mapping) and str(candidate.get("id")) == ident:
                    role = str(candidate.get("role", "")).upper()
                    if role in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
                        return role
                    spine = candidate.get("spine")
                    if isinstance(spine, Mapping):
                        role = str(spine.get("role", "")).upper()
                        if role in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
                            return role
        return "REPAIR"

    def _enrich_host_rows(self, rows: Sequence[object]) -> list[dict[str, object]]:
        """Copy host-owned role/spine facts onto model row presentation."""
        source_rows = {}
        if isinstance(self.host_draft, Mapping):
            source_rows = {
                str(item.get("row_id")): item for item in _items(self.host_draft.get("rows"))
                if isinstance(item, Mapping) and _nonempty_text(item.get("row_id"))
            }
        ledger = self._host_ledger()
        requirements = {
            str(item.get("id")): item for item in _items(ledger.get("requirements"))
            if isinstance(ledger, Mapping) and isinstance(item, Mapping)
        } if isinstance(ledger, Mapping) else {}
        enriched: list[dict[str, object]] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            row = copy.deepcopy(dict(raw))
            row_id = str(row.get("row_id", ""))
            host_row = source_rows.get(row_id, {})
            if isinstance(host_row, Mapping):
                for key in ("role", "witness_kind"):
                    if key in host_row:
                        row[key] = copy.deepcopy(host_row[key])
            requirement = requirements.get(row_id)
            if isinstance(requirement, Mapping) and isinstance(requirement.get("spine"), Mapping):
                row["spine"] = copy.deepcopy(dict(requirement["spine"]))
            enriched.append(row)
        return enriched

    def _s1_proposal_admission(
            self, proposal: object,
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        ledger = self._host_ledger()
        if not isinstance(ledger, Mapping):
            return None, {
                "decision": "UNKNOWN", "code": "HOST_LEDGER_REQUIRED",
                "unresolved": ["host_requirement_ledger"],
            }
        ledger = copy.deepcopy(dict(ledger))
        draft_rows = self.host_draft.get("rows", ()) if isinstance(self.host_draft, Mapping) else ()
        host_rows = {
            str(item.get("row_id")): item
            for item in _items(draft_rows) if isinstance(item, Mapping)
        }
        for requirement in _items(ledger.get("requirements")):
            if not isinstance(requirement, dict):
                continue
            host_row = host_rows.get(str(requirement.get("id")))
            role = str(host_row.get("role", "")).upper() if isinstance(host_row, Mapping) else ""
            if role in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
                requirement["role"] = role
        try:
            from kernel_l109_admission import admit_reviewed_proposal
            admitted, receipt = admit_reviewed_proposal(
                ledger, proposal,
                ticket_hash=self.ticket_hash,
                base_revision=self.base_revision,
                request_id=f"model-s1:{self.ticket_hash}:{self.epoch}",
                reviewer=self.semantic_reviewer,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return None, {
                "decision": "UNKNOWN", "code": "S1_PROPOSAL_INVALID",
                "unresolved": [type(error).__name__],
            }
        return admitted, receipt

    def _bind_admitted_ledger(self, ledger: Mapping[str, object]) -> None:
        self.requirement_ledger = copy.deepcopy(dict(ledger))
        try:
            from kernel_s1_spine import bind_idealpath_owner
            bind_idealpath_owner(self, ledger)
        except (KeyError, TypeError, ValueError, RuntimeError):
            # The admission call already validated the ledger. A separate
            # runtime projection may be unavailable in a light offline host;
            # retain the authoritative ledger and let later checks say why.
            self.requirement_table = None

    def _s1_row_coverage_matches_host(self, rows: Sequence[object]) -> bool:
        draft = self.host_draft
        if not isinstance(draft, Mapping):
            return True
        expected = {
            str(item.get("row_id")): frozenset(str(ref) for ref in _items(item.get("ticket_refs")))
            for item in _items(draft.get("rows")) if isinstance(item, Mapping)
        }
        actual = {
            str(item.get("row_id")): frozenset(str(ref) for ref in _items(item.get("ticket_refs")))
            for item in rows if isinstance(item, Mapping)
        }
        return bool(expected) and actual.keys() == expected.keys() and all(
            actual[key] == expected[key] for key in expected
        )

    def _admit_s2_force_draft(self) -> dict[str, object]:
        """Admit the host localization draft as the typed S2 -> S3 answer."""
        from kernel_s2_force import (
            build_receipt,
            build_s2_localization_draft,
            draft_target,
            enabled,
        )

        if not enabled():
            return self._reply(False, code="HOST_BLOCKED", missing=("typed_stage_exit",))
        draft = build_s2_localization_draft(
            causal_frontier=self.s2_force_frontier,
            model_evidence=self.stage_evidence.get("S2", ()),
            row_ids=tuple(self.rows),
            target_path=self.s2_force_target.get("path", ""),
            target_symbol=self.s2_force_target.get("symbol", ""),
            ticket_hash=self.ticket_hash,
            base_revision=self.base_revision,
        )
        answer = draft.get("answer")
        valid, missing = self._check_shape("S2", answer)
        if not valid or not isinstance(answer, Mapping):
            return self._reply(
                False, code="HOST_BLOCKED", missing=missing or ("s2_localization_draft",)
            )
        known, missing = self._known_rows(answer)
        if not known or _items(answer.get("unresolved_edges")):
            return self._reply(
                False, code="HOST_BLOCKED", missing=missing or ("resolved_localization_edges",)
            )
        target = draft_target(draft)
        if target is None:
            return self._reply(False, code="HOST_BLOCKED", missing=("localized_target",))
        self.s2_force_target = target
        previous_epoch = self.epoch
        self._advance("S3", answer)
        receipt = build_receipt(
            draft,
            stage_epoch=previous_epoch,
            ticket_hash=self.ticket_hash,
            base_revision=self.base_revision,
            forced=True,
            override=False,
        )
        self.s2_force_receipts.append(copy.deepcopy(receipt))
        self.s2_force_receipts = self.s2_force_receipts[-16:]
        result = self._reply(True, code="S2FORCE_ADMITTED", status="ADMITTED")
        result["s2_force_receipt"] = receipt
        result["s2_localization_draft"] = copy.deepcopy(draft)
        return result

    def _stage_exit_exhausted(self, *, runtime: object | None = None) -> dict[str, object]:
        """Stop safely when no typed late-stage answer was produced in budget."""
        if runtime is None:
            runtime = getattr(self, "_runtime", None)
        self.abstentions.append({
            "schema": "stateframe-stage-abstention/1",
            "stage": self.stage,
            "state_epoch": self.epoch,
            "provenance": "HOST_STAGE_EXIT_EXHAUSTED",
        })
        if runtime is not None:
            setattr(runtime, "system_outcome", "stalled")
        result = self._reply(
            False,
            code="STAGE_EXIT_EXHAUSTED",
            missing=("typed_stage_exit",),
        )
        result.update({
            "terminal": True,
            "disposition": "INCOMPLETE",
            "next_action": "host_stop",
            "admitted_actions": [],
            "continuation": {
                "schema": CONTRACT_SCHEMA,
                "owner": self.owner_id,
                "next_menu": [],
            },
            "stage_exit_budget": self.stage_exit_budget,
            "stage_turns_used": self.stage_turns.get(self.stage, 0),
        })
        return result

    def _admit_host_draft(
            self, arguments: Mapping[str, object] | None = None, *, host_forced: bool = False
    ) -> dict[str, object]:
        """Admit only the host draft after a typed abstention."""
        self.action_count += 1
        if self.action_count > self.max_actions:
            return self._reply(False, code="HOST_BLOCKED", missing=("budget",))
        if not host_forced:
            args = arguments if isinstance(arguments, Mapping) else {}
            if not {"stage", "state_epoch", "ticket_hash", "base_revision"}.issubset(args):
                return self._reply(False, code="INVALID_ANSWER", missing=("abstain_identity",))
            if args.get("stage") != self.stage or args.get("state_epoch") != self.epoch:
                return self._reply(False, code="STALE_EPOCH", missing=("active_stage_epoch",))
            if (
                args.get("ticket_hash") != self.ticket_hash
                or args.get("base_revision") != self.base_revision
            ):
                return self._reply(False, code="STALE_BASE", missing=("ticket_or_base_identity",))
        if self.stage in {"S4", "S5", "S6", "S7"} and self.forced_exit_action() is not None:
            return self._stage_exit_exhausted()
        if self.stage == "S2":
            forced = self._admit_s2_force_draft()
            self.clear_refusal_streak()
            return forced
        if self.stage != "S1":
            self.stage_turns[self.stage] = 0
            self._last_stage_turn_marker = None
            abstention = {
                "schema": "stateframe-stage-abstention/1",
                "stage": self.stage,
                "state_epoch": self.epoch,
                "provenance": "MODEL_ABSTAINED_HOST_STAGE_EXIT",
            }
            self.abstentions.append(abstention)
            return self._reply(False, code="HOST_BLOCKED", missing=("typed_stage_exit",))
        answer = self._host_draft_answer()
        if answer is None:
            return self._reply(False, code="HOST_BLOCKED", missing=("host_s1_draft",))
        spine_admission = self.host_draft.get("spine_admission") if isinstance(self.host_draft, Mapping) else None
        host_ledger = self.host_draft.get("requirement_ledger") if isinstance(self.host_draft, Mapping) else None
        if (
            not isinstance(spine_admission, Mapping)
            or spine_admission.get("status") != "ADMITTED"
            or not isinstance(host_ledger, Mapping)
            or not isinstance(host_ledger.get("spine"), Mapping)
        ):
            return self._reply(False, code="HOST_BLOCKED", missing=("s1_spine_semantic_admission",))
        valid, missing = self._check_shape("S1", answer)
        if not valid:
            return self._reply(False, code="HOST_BLOCKED", missing=missing)
        valid, missing = self._check_rows(answer)
        if not valid:
            return self._reply(False, code="HOST_BLOCKED", missing=missing)
        self.rows = {
            str(row["row_id"]): dict(copy.deepcopy(dict(row)))
            for row in self._enrich_host_rows(_items(answer["rows"]))
            if isinstance(row, Mapping)
        }
        if isinstance(host_ledger, Mapping):
            self._bind_admitted_ledger(host_ledger)
        draft_review = self.host_draft.get("semantic_review") if isinstance(self.host_draft, Mapping) else None
        if isinstance(draft_review, Mapping):
            self.semantic_review_receipt = copy.deepcopy(dict(draft_review))
        receipt = self._register_requirement_table(
            answer, provenance="HOST_DRAFT_ADMITTED_AFTER_ABSTAIN"
        )
        self._advance("S2", answer)
        return self._reply(
            True,
            code="S1_ADMITTED",
            status="ADMITTED",
            requirement_table_receipt=receipt,
        )

    def _actions(self) -> tuple[str, ...]:
        forced = self._forced_actions()
        if forced:
            return forced
        if self.terminal:
            return (TOOL_NAME, "done", "inspect", "read")
        actions = [TOOL_NAME, "inspect", "read"]
        if self.stage == "S1":
            actions.insert(0, S1_COMPLETION_TOOL_NAME)
        if self.stage in {"S1", "S2"}:
            actions.append(LOCALIZE_TOOL_NAME)
        if self.stage == "S2":
            actions.append(CAUSAL_LOCALIZE_TOOL_NAME)
        if self.stage == "S3":
            actions.insert(1, TESTWRITER_TOOL_NAME)
            actions.insert(2, "typed_check_spec")
        if self.stage in BACKTRACK_TARGETS:
            actions.append("backtrack")
        return tuple(actions)

    def capability_set(self) -> dict[str, object]:
        """Return the current stage's admission set for every action class."""
        actions = self._actions()
        allowed = set(actions) if self._forced_actions() else set(STAGE_ACTIONS[self.stage])
        allowed.update(actions)
        if self.stage in BACKTRACK_TARGETS:
            allowed.add("backtrack")
        effects: set[str] = set()
        if allowed & SOURCE_READ_ACTIONS:
            effects.add("SOURCE_READ")
        if allowed & DIAGNOSTIC_ACTIONS:
            effects.add("SCRATCH_EXECUTE")
        if allowed & PRODUCT_WRITE_ACTIONS:
            effects.add("PRODUCT_WRITE")
        if "done" in allowed:
            effects.add("CERTIFY")
        return {
            "schema": "stateframe-stage-capability/1",
            "stage": self.stage,
            "epoch": self.epoch,
            "base_digest": self.base_revision,
            "satisfied_predicate_refs": sorted(self.answers),
            "allowed_actions": sorted(allowed),
            "allowed_effects": sorted(effects),
            "pending_predicate_ids": [f"{self.stage}:answer"],
            "recovery_action_ids": [
                action for action in (
                    *sorted(STAGE_EXIT_ACTIONS[self.stage]), TOOL_NAME, "inspect", "read", "probe",
                    "run_tests", "typed_check_spec", TESTWRITER_TOOL_NAME,
                    CAUSAL_LOCALIZE_TOOL_NAME, S2_FORCE_ALTERNATIVE_ACTION, ABSTAIN_ACTION,
                )
                if action in allowed
            ],
            "framebuild_receipt": self.framebuild_receipt(),
        }

    def stage_exit_liveness(self, offered_actions: Sequence[object] | None = None) -> dict[str, object]:
        """Evaluate the GENERAL exit-action invariant for this stage."""
        offered = self._actions() if offered_actions is None else offered_actions
        return stage_exit_liveness(self.stage, offered, self.capability_set()["allowed_actions"])

    def assert_stage_exit_liveness(self, offered_actions: Sequence[object] | None = None) -> dict[str, object]:
        """Fail closed when a projected menu loses its only stage exit."""
        offered = self._actions() if offered_actions is None else offered_actions
        return assert_stage_exit_liveness(
            self.stage, offered, self.capability_set()["allowed_actions"]
        )

    def menu_admissible_actions(self) -> frozenset[str]:
        """MENUADMIT: the ONE action set both menu projection and dispatch use.

        A projected menu may only offer actions in this set, and dispatch
        admission accepts exactly this set.  Under an exhausted stage budget it
        is the forced exit slice (never a source read), so a stale navigation
        action can no longer be offered and then force-closed at dispatch.
        """
        forced = self._forced_actions()
        if forced:
            return frozenset(forced)
        allowed = set(self.capability_set()["allowed_actions"])
        if self.stage == "S3":
            # The live S3 carrier is intentionally the verified
            # TestWriter path; legacy typed_check_spec remains callable by
            # host compatibility probes but is not model-admissible here.
            # backtrack stays admissible: the S3 menu offers it as the typed
            # recovery route, so dispatch must not refuse what was offered.
            allowed &= {TOOL_NAME, TESTWRITER_TOOL_NAME, "backtrack"}
        return frozenset(allowed)

    def dispatch_admits(self, action: object) -> bool:
        return str(action or "").strip().lower() in self.menu_admissible_actions()

    def forced_host_fallback(self, action: object) -> bool:
        """True when an admitted forced-menu action is executed as the host fallback.

        With S3FORCE on, question_gate/abstain in the exhausted S3 menu are
        legitimate choices that the host turns into its own forced exit;
        typed_test_writer stays the model-executed override.
        """
        name = str(action or "").strip().lower()
        if self.stage != "S3" or self.forced_exit_action() is None:
            return False
        from kernel_s3_force import enabled
        return bool(enabled()) and name in {self.forced_exit_action(), ABSTAIN_ACTION}

    def admit_action(self, action: object) -> dict[str, object]:
        """Apply stage admission to ordinary and reducer-owned dispatches."""
        name = str(action or "").strip().lower()
        capabilities = self.capability_set()
        allowed = self.dispatch_admits(name)
        return {
            "schema": "stateframe-stage-admission/1",
            "action": name,
            "stage": self.stage,
            "epoch": self.epoch,
            "admitted": allowed,
            "reason_code": "STAGE_ADMITTED" if allowed else "STAGE_CAPABILITY_REJECTED",
            "capability_set": capabilities,
        }

    def menu(self) -> dict[str, object]:
        """Build the exact menu consumed by schema, dispatch, and continuation."""
        actions = self._actions()
        typed_check_tool = None
        typed_test_writer_tool = None
        causal_localize_tool = None
        if self.stage == "S3":
            from kernel_reprospec import typed_check_spec_schema
            from kernel_testwriter import typed_test_writer_schema
            typed_check_tool = typed_check_spec_schema({"modules": []})
            typed_test_writer_tool = typed_test_writer_schema({"modules": []}, owner=self)
        if self.stage == "S2":
            from kernel_causal_localizer import tool_schema as causal_tool_schema
            causal_localize_tool = causal_tool_schema()
        return {
            "schema": CONTRACT_SCHEMA,
            "owner": self.owner_id,
            "stage": self.stage,
            "state_epoch": self.epoch,
            "ticket_hash": self.ticket_hash,
            "base_revision": self.base_revision,
            "actions": list(actions),
            # ``tool`` keeps the IDEALPATH1 public all-stage contract.  The
            # live runtime uses ``current_tool`` below so its admitted menu is
            # narrowed to the reducer's active stage without breaking the
            # existing contract consumer.
            "tool": copy.deepcopy(tool_schema()) if TOOL_NAME in actions else None,
            "current_tool": copy.deepcopy(stage_tool_schema(self.stage)) if TOOL_NAME in actions else None,
            "requirement_table_tool": (
                copy.deepcopy(requirement_table_tool_schema(self.host_draft))
                if S1_COMPLETION_TOOL_NAME in actions else None
            ),
            "abstain_tool": (
                self._abstain_tool_schema()
                if ABSTAIN_ACTION in actions else None
            ),
            "localization_alternative_tool": (
                self._s2_force_alternative_tool_schema()
                if self.stage == "S2" and S2_FORCE_ALTERNATIVE_ACTION in actions else None
            ),
            "backtrack_tool": (
                copy.deepcopy(backtrack_tool_schema(BACKTRACK_TARGETS[self.stage]))
                if "backtrack" in actions else None
            ),
            "typed_check_tool": typed_check_tool,
            "typed_test_writer_tool": typed_test_writer_tool,
            "localize_tool": (copy.deepcopy(localize_tool_schema())
                               if LOCALIZE_TOOL_NAME in actions else None),
            "causal_localize_tool": causal_localize_tool,
            "answer_fields": list(
                (*_ANSWER_FIELDS[self.stage], *_OPTIONAL_ANSWER_FIELDS.get(self.stage, ()))
            ) if TOOL_NAME in actions else [],
            "backtrack_targets": sorted(BACKTRACK_TARGETS.get(self.stage, ())),
            "capability_set": self.capability_set(),
            "stage_budget": self.stage_budget(),
            "s2_force_menu_notice": self.s2_force_menu_notice(),
            "s3_force_menu_notice": self.s3_force_menu_notice(),
            "framebuild_receipt": self.framebuild_receipt(),
            "host_draft": copy.deepcopy(self.host_draft) if self.stage == "S1" else None,
            "stage_evidence": copy.deepcopy(self.stage_evidence.get(self.stage, ())),
            "suspects": (
                copy.deepcopy(self.stage_evidence.get("S2", ()))
                if self.stage == "S2" else []
            ),
            "result_fields": ["result", "freshness", "certification"],
            "capability_snapshot": (
                self._capability_snapshot.to_dict()
                if self._capability_snapshot is not None else None
            ),
        }

    def _abstain_tool_schema(self) -> dict[str, object]:
        from kernel_s1_force import abstain_tool_schema
        return abstain_tool_schema(
            self.stage,
            state_epoch=self.epoch,
            ticket_hash=self.ticket_hash,
            base_revision=self.base_revision,
        )

    def _s2_force_alternative_tool_schema(self) -> dict[str, object]:
        from kernel_s2_force import alternative_tool_schema
        return alternative_tool_schema(
            state_epoch=self.epoch,
            ticket_hash=self.ticket_hash,
            base_revision=self.base_revision,
        )

    def continuation(self) -> dict[str, object]:
        """Return the only continuation contract after a dispatch result."""
        return {
            "schema": CONTRACT_SCHEMA,
            "owner": self.owner_id,
            "next_menu": self.menu(),
        }

    def _reply(
            self,
            accepted: bool,
            *,
            code: str,
            missing: Sequence[str] = (),
            admitted_actions: Sequence[str] | None = None,
            status: str = "REJECTED",
            requirement_table_receipt: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        actions = tuple(admitted_actions) if admitted_actions is not None else self._actions()
        reply = {
            "schema": SCHEMA,
            "accepted": accepted,
            "status": status if accepted or status != "REJECTED" else "REJECTED",
            "stage": self.stage,
            "state_epoch": self.epoch,
            "missing_predicates": list(dict.fromkeys(str(item) for item in missing)),
            "reason_code": code if code in REASON_CODES else "CONTRACT_REJECTED",
            "admitted_actions": list(actions),
            "remaining_budget": max(0, self.max_actions - self.action_count),
            "continuation": self.continuation(),
        }
        if requirement_table_receipt is not None:
            reply["requirement_table_receipt"] = copy.deepcopy(dict(requirement_table_receipt))
        return reply

    def _check_shape(self, stage: str, answer: object) -> tuple[bool, tuple[str, ...]]:
        if not isinstance(answer, Mapping):
            return False, ("answer_object",)
        required = set(_ANSWER_FIELDS[stage])
        allowed = required | set(_OPTIONAL_ANSWER_FIELDS.get(stage, ()))
        if not required.issubset(set(answer)) or not set(answer).issubset(allowed):
            return False, ("answer_fields",)
        if any(key in answer for key in ("prose", "settlement", "natural_language", "freeform")):
            return False, ("menu_only_answer",)
        for key, value in answer.items():
            if key in {
                "rows", "edges", "unresolved_edges", "hunks", "causal_links",
                "risk_dispositions", "changes",
            }:
                if not _items(value) and not (key == "unresolved_edges" and stage == "S2"):
                    return False, (key,)
            elif key in {
                "row_ids", "entry_refs", "decision_refs", "outcome_refs", "product_refs",
                "fixtures", "monitors", "preserved_invariants", "risks",
                "regression_test_refs", "regression_refs", "contract_changes", "predicted_checks",
                "scope_exceptions", "quality_check_refs", "unresolved_row_ids",
                "residual_risks", "limitations", "verification_refs",
            }:
                if not _refs(
                    value,
                    allow_empty=stage == "S7" and key in {"unresolved_row_ids", "limitations"},
                ):
                    return False, (key,)
            elif key in {
                "scenario", "oracle", "entry_hint", "assumptions", "alternative",
                "proposed_change", "wrong_repair_control", "sanity_control",
                "cause_proposal", "edit_proposal", "verification_manifest", "finish_proposal",
                "focus", "ticket_focus", "proposal",
            }:
                if _mapping(value) is None and not _items(value):
                    return False, (key,)
            elif not _nonempty_text(value) and key not in {"base_revision"}:
                return False, (key,)
        return True, ()

    def _check_rows(self, answer: Mapping[str, object]) -> tuple[bool, tuple[str, ...]]:
        raw_rows = _items(answer.get("rows"))
        if not raw_rows:
            return False, ("rows",)
        seen: set[str] = set()
        for row in raw_rows:
            if not isinstance(row, Mapping):
                return False, ("row_object",)
            required = {
                "row_id", "ticket_refs", "trigger", "observed_symptom",
                "expected_observable", "constraints", "expectation_refs",
                "scenario_provenance", "oracle_provenance", "ambiguity_status",
            }
            allowed = required | {
                "wrong_behavior", "spine", "focus", "ticket_focus", "role", "witness_kind",
            }
            if not required.issubset(set(row)) or not set(row).issubset(allowed):
                return False, ("row_fields",)
            row_id = _text(row.get("row_id"), limit=80)
            if row_id in seen or not row_id or not _refs(row.get("ticket_refs")):
                return False, ("row_identity",)
            ticket_refs = {str(item) for item in _items(row.get("ticket_refs"))}
            if self.ticket_spans and not ticket_refs.issubset(self.ticket_spans):
                return False, ("ticket_provenance",)
            if not all(
                _nonempty_text(row.get(key))
                for key in ("trigger", "observed_symptom", "expected_observable")
            ):
                return False, ("scenario_oracle_text",)
            if "wrong_behavior" in row and not _nonempty_text(row.get("wrong_behavior")):
                return False, ("wrong_behavior",)
            if "spine" in row:
                spine = _mapping(row.get("spine"))
                if spine is None or spine.get("semantic_admission") != "ADMITTED":
                    return False, ("spine_admission",)
            if not _refs(row.get("expectation_refs"), allow_empty=True) or not _refs(
                row.get("constraints"), allow_empty=True
            ):
                return False, ("row_arrays",)
            for key in ("scenario_provenance", "oracle_provenance"):
                provenance = _mapping(row.get(key))
                if (
                    provenance is None
                    or set(provenance) != {"kind", "refs"}
                    or not _nonempty_text(provenance.get("kind"))
                    or not _refs(provenance.get("refs"))
                ):
                    return False, (key,)
            if str(row.get("ambiguity_status", "")).upper() not in {"NONE", "RESOLVED"}:
                return False, ("ambiguity_status",)
            seen.add(row_id)
        if self._spine_row_refs and seen != {ident for ident, _version in self._spine_row_refs}:
            return False, ("canonical_spine_row_set",)
        return True, ()

    def _known_rows(self, answer: Mapping[str, object]) -> tuple[bool, tuple[str, ...]]:
        raw_ids = answer.get("row_ids")
        if self.stage == "S6":
            raw_ids = [
                row.get("row_id")
                for row in _items(answer.get("rows"))
                if isinstance(row, Mapping)
            ]
        ids = tuple(str(item) for item in _items(raw_ids))
        known = {ident for ident, _version in self._spine_row_refs} if self._spine_row_refs else set(self.rows)
        missing = tuple(item for item in ids if item not in known)
        return not missing and bool(ids), tuple(f"unknown_row:{item}" for item in missing)

    def _advance(self, next_stage: str, answer: Mapping[str, object]) -> None:
        previous = self.stage
        self.answers[previous] = dict(copy.deepcopy(dict(answer)))
        if previous == "S1":
            self.stage_evidence["S2"] = copy.deepcopy(self.navigation_evidence[-64:])
        self.stage = next_stage
        self.epoch += 1
        self._capability_snapshot = None
        self._stale_batch_gate = None
        self.stage_turns[next_stage] = 0
        self._last_stage_turn_marker = None
        self.transitions.append({
            "from": previous, "to": next_stage, "state_epoch": self.epoch, "kind": "forward",
        })

    def _register_requirement_table(
            self, answer: Mapping[str, object], *, provenance: str = "MODEL_TYPED_DECISION"
    ) -> dict[str, object]:
        """Mint the host receipt that makes the S1 exit observable and durable."""
        rows = [
            copy.deepcopy(dict(row))
            for row in _items(answer.get("rows"))
            if isinstance(row, Mapping)
        ]
        span_refs = sorted({
            str(reference)
            for row in rows
            for reference in _items(row.get("ticket_refs"))
        })
        table_hash = _digest({"ticket_hash": self.ticket_hash, "rows": rows})
        receipt = {
            "schema": "stateframe-idealpath-requirement-table-receipt/1",
            "receipt_id": "idealpath-s1-table:" + table_hash[7:31],
            "ticket_hash": self.ticket_hash,
            "base_revision": self.base_revision,
            "table_hash": table_hash,
            "row_ids": [str(row["row_id"]) for row in rows],
            "ticket_span_refs": span_refs,
            "status": "ADMITTED",
            "provenance": provenance,
        }
        if isinstance(self.semantic_review_receipt, Mapping):
            receipt["semantic_review_receipt"] = copy.deepcopy(dict(self.semantic_review_receipt))
        self.requirement_table_receipt = receipt
        return copy.deepcopy(receipt)

    def submit(self, request: Mapping[str, object]) -> dict[str, object]:
        """Accept one typed answer; host metadata is checked, never model-owned."""
        self.action_count += 1
        if self.action_count > self.max_actions:
            return self._reply(False, code="HOST_BLOCKED", missing=("budget",))
        if request.get("stage") != self.stage or request.get("state_epoch") != self.epoch:
            return self._reply(False, code="STALE_EPOCH", missing=("active_stage_epoch",))
        if (
            request.get("ticket_hash") != self.ticket_hash
            or request.get("base_revision") != self.base_revision
        ):
            return self._reply(False, code="STALE_BASE", missing=("ticket_or_base_identity",))
        if any(field in request for field in ("capability_digest", "budget_generation")):
            snapshot = self._capability_snapshot
            if (
                snapshot is None
                or request.get("capability_digest") != snapshot.capability_digest
                or request.get("budget_generation") != snapshot.budget_generation
            ):
                return self._reply(False, code="STALE_REPLAN", missing=("frozen_menu_snapshot",))
        answer = request.get("answer")
        valid, missing = self._check_shape(self.stage, answer)
        if not valid:
            return self._reply(False, code="INVALID_ANSWER", missing=missing)
        assert isinstance(answer, Mapping)
        if self.stage == "S1":
            valid, missing = self._check_rows(answer)
            if not valid:
                return self._reply(False, code="INVALID_ANSWER", missing=missing)
            raw_rows = _items(answer.get("rows"))
            if not self._s1_row_coverage_matches_host(raw_rows):
                return self._reply(False, code="INVALID_ANSWER", missing=("host_row_coverage",))
            if any(
                    isinstance(row, Mapping) and "spine" in row
                    for row in raw_rows):
                return self._reply(
                    False,
                    code="INVALID_ANSWER",
                    missing=("host_owned_s1_spine",),
                )
            host_ledger = self._host_ledger()
            has_host_spine = isinstance(host_ledger, Mapping) and isinstance(
                host_ledger.get("spine"), Mapping)
            if not has_host_spine:
                proposal = answer.get("proposal")
                if proposal is None:
                    return self._reply(
                        False, code="HOST_BLOCKED",
                        missing=("s1_semantic_proposal", "host_semantic_review"),
                    )
                admitted, review = self._s1_proposal_admission(proposal)
                self.semantic_review_receipt = copy.deepcopy(review)
                decision = str(review.get("decision", "UNKNOWN")).upper()
                if admitted is None or decision != "SUPPORTED":
                    code = (
                        "S1_SEMANTIC_REVIEW_CONTRADICTED"
                        if decision == "CONTRADICTED" else "S1_SEMANTIC_REVIEW_UNKNOWN"
                    )
                    unresolved = tuple(str(item) for item in (
                        _items(review.get("unresolved")) or ("semantic_admission",)
                    ))
                    return self._reply(False, code=code, missing=unresolved) | {
                        "semantic_review": copy.deepcopy(review),
                    }
                self._bind_admitted_ledger(admitted)
                if isinstance(self.host_draft, Mapping):
                    self.host_draft = copy.deepcopy(dict(self.host_draft))
                    self.host_draft["requirement_ledger"] = copy.deepcopy(dict(admitted))
                    self.host_draft["spine_admission"] = {
                        "status": "ADMITTED", "code": "S1SPINE_ADMITTED",
                        "provenance": "HOST_MODEL_PROPOSAL_REVIEW",
                    }
                    self.host_draft["semantic_review"] = copy.deepcopy(review)
            enriched_rows = self._enrich_host_rows(raw_rows)
            answer = dict(copy.deepcopy(dict(answer)))
            answer["rows"] = enriched_rows
            self.rows = {
                str(row["row_id"]): dict(copy.deepcopy(dict(row)))
                for row in enriched_rows
                if isinstance(row, Mapping)
            }
            self.ticket_focus = copy.deepcopy(answer.get("focus", answer.get("ticket_focus")))
            requirement_table_receipt = self._register_requirement_table(
                answer, provenance=(
                    "MODEL_TYPED_DECISION_WITH_HOST_SPINE"
                    if self.semantic_review_receipt else "MODEL_TYPED_DECISION"
                ),
            )
            self._advance("S2", answer)
            return self._reply(
                True,
                code="S1_ADMITTED",
                status="ADMITTED",
                requirement_table_receipt=requirement_table_receipt,
            )
        if self.stage in {"S2", "S3", "S4", "S6"}:
            known, missing = self._known_rows(answer)
            if not known:
                return self._reply(False, code="UNKNOWN_ROW", missing=missing)
        if self.stage == "S2":
            if _items(answer.get("unresolved_edges")):
                return self._reply(False, code="INVALID_ANSWER", missing=("unresolved_edges",))
            self._advance("S3", answer)
            return self._reply(True, code="S2_ADMITTED", status="ADMITTED")
        if self.stage == "S3":
            missing_evidence = tuple(
                row_id for row_id in self.rows
                if (
                    self.row_role(row_id) == "PRESERVE"
                    and row_id not in self.admitted_parent_passes
                ) or (
                    self.row_role(row_id) != "PRESERVE"
                    and row_id not in self.admitted_failures
                )
            )
            if missing_evidence:
                return self._reply(
                    False,
                    code="MISSING_RECEIPT",
                    missing=tuple(
                        f"admitted_parent_pass:{item}" if self.row_role(item) == "PRESERVE"
                        else f"admitted_failure:{item}" for item in missing_evidence
                    ),
                )
            self._advance("S4", answer)
            return self._reply(True, code="S3_ADMITTED", status="ADMITTED")
        if self.stage == "S4":
            alternative = _mapping(answer.get("alternative"))
            if alternative is None or str(alternative.get("disposition", "")).upper() != "CONTRADICTED":
                return self._reply(
                    False,
                    code="INVALID_ANSWER",
                    missing=("nearest_alternative_contradicted",),
                )
            typed_cause = answer.get("cause_proposal")
            if typed_cause is not None:
                if not isinstance(typed_cause, Mapping):
                    return self._reply(False, code="INVALID_ANSWER", missing=("cause_proposal",))
                admission = self.admit_s47_cause(typed_cause)
                if admission.get("status") != "SUPPORTED":
                    return self._reply(
                        False,
                        code="HOST_BLOCKED",
                        missing=tuple(admission.get("missing_predicate_ids", ())) or ("cause_admission",),
                    )
            self._advance("S5", answer)
            return self._reply(True, code="S4_ADMITTED", status="ADMITTED")
        if self.stage == "S5":
            if not self._patch_hunks_bound(answer):
                return self._reply(
                    False, code="INVALID_ANSWER", missing=("cause_to_hunk_binding",)
                )
            self._advance("S6", answer)
            return self._reply(True, code="S5_ADMITTED", status="ADMITTED")
        if self.stage == "S6":
            typed_manifest = answer.get("verification_manifest")
            if typed_manifest is not None:
                if not isinstance(typed_manifest, Mapping):
                    return self._reply(False, code="INVALID_ANSWER", missing=("verification_manifest",))
                decision = self.verify_s47_manifest(
                    typed_manifest,
                    self.s47_verification_context,
                )
                if decision.get("status") == "READY_FOR_S7":
                    self.verification = {
                        "certified": True,
                        "s47_decision": copy.deepcopy(dict(decision)),
                    }
            if not self._verification_ready():
                return self._reply(
                    False,
                    code="CERTIFICATION_REQUIRED",
                    missing=("current_verification_conjunction",),
                )
            self._advance("S7", answer)
            return self._reply(True, code="S6_ADMITTED", status="ADMITTED")
        outcome = str(answer.get("outcome", ""))
        if outcome == "VERIFIED_FIXED" and not self._verification_ready():
            return self._reply(
                False, code="CERTIFICATION_REQUIRED", missing=("verified_fixed_conjunction",)
            )
        self.answers["S7"] = dict(copy.deepcopy(dict(answer)))
        return self._reply(
            True,
            code="S7_ADMITTED",
            status="CERTIFIED" if outcome == "VERIFIED_FIXED" else "INCOMPLETE",
        )

    def _patch_hunks_bound(self, answer: Mapping[str, object]) -> bool:
        hunks = _items(answer.get("hunks"))
        return bool(hunks) and all(
            isinstance(hunk, Mapping)
            and _nonempty_text(hunk.get("hunk_ref"))
            and _nonempty_text(hunk.get("cause_or_regression_ref"))
            for hunk in hunks
        )

    def record_check(
            self,
            *,
            requirement_id: str,
            status: object,
            product_revision: str,
            current_revision: str,
            check_hash: str,
            stdout: object = "",
            stderr: object = "",
            reached_observation: object = None,
            failure_phase: object = None,
            outcome: object = None,
            ticket_relevant: bool = False,
    ) -> dict[str, object]:
        """Mint a receipt with independent result, freshness, and certification."""
        receipt_id = "idealpath-check:" + _digest({
            "requirement_id": requirement_id,
            "status": status,
            "product_revision": product_revision,
            "current_revision": current_revision,
            "check_hash": check_hash,
            "stdout": stdout,
            "stderr": stderr,
        })[7:31]
        existing = self.receipts.get(receipt_id)
        if existing is not None:
            cached = copy.deepcopy(existing)
            cached["cache_hit"] = True
            return cached
        receipt = _typed_receipt(
            receipt_id=receipt_id,
            requirement_id=requirement_id,
            status=status,
            product_revision=product_revision,
            current_revision=current_revision,
            check_hash=check_hash,
            stdout=stdout,
            stderr=stderr,
            reached_observation=reached_observation,
            failure_phase=failure_phase,
            outcome=outcome,
            ticket_relevant=ticket_relevant,
        )
        self.receipts[receipt_id] = receipt
        self.settle_stage_action(
            f"receipt:{receipt_id}", action="check", receipt_ids=(receipt_id,),
        )
        return copy.deepcopy(receipt)

    def admit_ticket_failure(self, receipt_id: str, *, requirement_id: str) -> dict[str, object]:
        """Credit a REPAIR/NEW_BEHAVIOR row for a reached parent failure."""
        return self.admit_ticket_result(receipt_id, requirement_id=requirement_id)

    def admit_ticket_result(self, receipt_id: str, *, requirement_id: str) -> dict[str, object]:
        """Admit role-specific parent evidence without accepting model role text."""
        receipt = self.receipts.get(receipt_id)
        if requirement_id not in self.rows or receipt is None:
            return self._reply(False, code="UNKNOWN_ROW", missing=("requirement_or_receipt",))
        result = receipt["result"]
        freshness = receipt["freshness"]
        assert isinstance(result, Mapping) and isinstance(freshness, Mapping)
        if receipt.get("requirement_id") != requirement_id:
            return self._reply(False, code="MISSING_RECEIPT", missing=("requirement_receipt_match",))
        if result.get("outcome") == "ENVIRONMENT":
            receipt["certification"] = {
                "status": "REJECTED", "admitted": False, "basis": "ENVIRONMENT_NOT_REPRODUCED",
            }
            return self._reply(
                False,
                code="ENVIRONMENT_NOT_REPRODUCED",
                missing=("ticket_relevant_behavior_failure",),
            )
        role = self.row_role(requirement_id)
        expected = "PASS" if role == "PRESERVE" else "BEHAVIOR_FAIL"
        if (result.get("outcome") != expected
                or result.get("reached_observation") is not True
                or not result.get("ticket_relevant")):
            return self._reply(False, code="MISSING_RECEIPT", missing=(
                "reached_parent_pass" if role == "PRESERVE" else "reached_ticket_failure",
            ))
        if (
            freshness.get("product_revision") != self.base_revision
            or freshness.get("is_current") is not True
        ):
            return self._reply(False, code="STALE_BASE", missing=("untouched_revision",))
        receipt["certification"] = {
            "status": "ADMITTED", "admitted": True,
            "basis": "ticket_relevant_parent_pass_on_untouched_revision"
            if role == "PRESERVE" else "ticket_relevant_failure_on_untouched_revision",
            "role": role,
        }
        if role == "PRESERVE":
            self.admitted_parent_passes[requirement_id] = receipt_id
        else:
            self.admitted_failures[requirement_id] = receipt_id
        return self._reply(True, code="S3_ADMITTED", status="ADMITTED")

    def record_authored_witness(self, *, requirement_id: str, receipt_id: str) -> bool:
        """Record host-authored provenance for existing-entry NEW_BEHAVIOR."""
        receipt = self.receipts.get(str(receipt_id))
        if self.row_role(requirement_id) != "NEW_BEHAVIOR" or not isinstance(receipt, Mapping):
            return False
        result = receipt.get("result")
        freshness = receipt.get("freshness")
        if not isinstance(result, Mapping) or not isinstance(freshness, Mapping):
            return False
        if (result.get("outcome") != "BEHAVIOR_FAIL"
                or result.get("reached_observation") is not True
                or freshness.get("integrity") is not True):
            return False
        self.authored_witnesses[str(requirement_id)] = str(receipt_id)
        return True

    def record_verification(
            self,
            *,
            candidate_revision: str,
            pairs: Mapping[str, Mapping[str, object]],
            regressions: Sequence[str],
            wrong_repair_viable: bool,
            wrong_repair_rejected: bool,
            integrity: bool,
    ) -> dict[str, object]:
        """Record the S6 conjunction without allowing a score to certify it."""
        pair_ok = (
            isinstance(pairs, Mapping)
            and bool(pairs)
            and set(str(item) for item in pairs) == set(self.rows)
        )
        repair_rows: list[str] = []
        preserve_rows: list[str] = []
        obligation_results: dict[str, dict[str, object]] = {}
        for row_id, pair in pairs.items() if isinstance(pairs, Mapping) else ():
            if not isinstance(pair, Mapping):
                pair_ok = False
                continue
            baseline = self.receipts.get(str(pair.get("baseline", "")))
            candidate = self.receipts.get(str(pair.get("candidate", "")))
            if row_id not in self.rows or baseline is None or candidate is None:
                pair_ok = False
                continue
            base_result = baseline["result"]
            cand_result = candidate["result"]
            base_freshness = baseline["freshness"]
            cand_freshness = candidate["freshness"]
            assert isinstance(base_result, Mapping) and isinstance(cand_result, Mapping)
            assert isinstance(base_freshness, Mapping) and isinstance(cand_freshness, Mapping)
            role = self.row_role(row_id)
            declared_role = str(pair.get("role", role)).upper()
            if declared_role != role:
                pair_ok = False
            if role in {"REPAIR", "NEW_BEHAVIOR"}:
                repair_rows.append(str(row_id))
                valid_pair = (
                    base_result.get("outcome") == "BEHAVIOR_FAIL"
                    and base_result.get("reached_observation") is True
                    and base_freshness.get("product_revision") == self.base_revision
                    and base_freshness.get("is_current") is True
                    and base_freshness.get("integrity") is True
                    and cand_result.get("outcome") == "PASS"
                    and cand_result.get("reached_observation") is True
                    and cand_freshness.get("product_revision") == candidate_revision
                    and cand_freshness.get("is_current") is True
                    and cand_freshness.get("integrity") is True
                    and base_freshness.get("check_hash") == cand_freshness.get("check_hash")
                    and self.admitted_failures.get(str(row_id)) == str(pair.get("baseline", ""))
                )
                if role == "NEW_BEHAVIOR":
                    witness_ref = self.authored_witnesses.get(str(row_id))
                    valid_pair = valid_pair and bool(witness_ref)
            elif role == "PRESERVE":
                preserve_rows.append(str(row_id))
                valid_pair = (
                    base_result.get("outcome") == "PASS"
                    and base_result.get("reached_observation") is True
                    and base_freshness.get("product_revision") == self.base_revision
                    and base_freshness.get("is_current") is True
                    and base_freshness.get("integrity") is True
                    and cand_result.get("outcome") == "PASS"
                    and cand_result.get("reached_observation") is True
                    and cand_freshness.get("product_revision") == candidate_revision
                    and cand_freshness.get("is_current") is True
                    and cand_freshness.get("integrity") is True
                    and base_freshness.get("check_hash") == cand_freshness.get("check_hash")
                    and self.admitted_parent_passes.get(str(row_id)) == str(pair.get("baseline", ""))
                )
            else:
                valid_pair = False
            obligation_results[str(row_id)] = {
                "role": role,
                "certified": bool(valid_pair and integrity and declared_role == role),
            }
            pair_ok = pair_ok and valid_pair and declared_role == role
            frozen = self.frozen_repros.get(str(row_id))
            if isinstance(frozen, Mapping):
                pair_ok = pair_ok and (
                    base_freshness.get("check_hash") == frozen.get("check_hash")
                    and cand_freshness.get("check_hash") == frozen.get("check_hash")
                )
        regressions_ok = bool(regressions) and all(
            str(item).upper() in {"PASS", "PREEXISTING_FAILURE"} for item in regressions
        )
        self.verification = {
            "candidate_revision": candidate_revision,
            "pairs": dict(copy.deepcopy(dict(pairs))),
            "regressions": list(regressions),
            "pair_ok": pair_ok,
            "repair_rows": repair_rows,
            "preserve_rows": preserve_rows,
            "obligations": obligation_results,
            "regressions_ok": regressions_ok,
            "wrong_repair_viable": wrong_repair_viable,
            "wrong_repair_rejected": wrong_repair_rejected,
            "integrity": integrity,
            "certified": (
                pair_ok
                and regressions_ok
                and wrong_repair_viable
                and wrong_repair_rejected
                and integrity
            ),
        }
        certified = bool(self.verification["certified"])
        return self._reply(
            certified,
            code="S6_ADMITTED" if certified else "CERTIFICATION_REQUIRED",
            missing=() if certified else ("verification_conjunction",),
            status="CERTIFIED" if certified else "REJECTED",
        )

    def resolve_regression_refs(self, refs: Sequence[object]) -> tuple[str, ...]:
        """Resolve host receipt IDs to reducer statuses before comparison."""
        resolved: list[str] = []
        for raw in refs:
            if isinstance(raw, Mapping):
                candidate = raw.get("receipt_id", raw.get("id"))
                if not isinstance(candidate, str):
                    continue
                value = candidate
            else:
                value = str(raw)
            upper = value.upper()
            if upper in {"PASS", "PREEXISTING_FAILURE"}:
                resolved.append(upper)
                continue
            receipt = self.receipts.get(value)
            result = receipt.get("result") if isinstance(receipt, Mapping) else None
            freshness = receipt.get("freshness") if isinstance(receipt, Mapping) else None
            if not isinstance(result, Mapping) or not isinstance(freshness, Mapping):
                continue
            if result.get("outcome") == "PASS" and freshness.get("is_current") is True:
                resolved.append("PASS")
            elif (result.get("outcome") == "BEHAVIOR_FAIL"
                  and freshness.get("product_revision") == self.base_revision
                  and freshness.get("is_current") is True):
                resolved.append("PREEXISTING_FAILURE")
        return tuple(resolved)

    def _verification_ready(self) -> bool:
        return bool(
            isinstance(self.verification, Mapping)
            and self.verification.get("certified") is True
        )

    def completion_predicates(self) -> dict[str, object]:
        """Expose only host-derived S6/S7 predicates to the completion owner."""
        s6_ready = self._verification_ready()
        s7_answer = self.answers.get("S7")
        s7_admitted = isinstance(s7_answer, Mapping) and bool(s7_answer)
        return {
            "schema": SCHEMA,
            "stage": self.stage,
            "state_epoch": self.epoch,
            "s6_verification_ready": s6_ready,
            "s7_answer_admitted": s7_admitted,
            "completion_admitted": (
                self.stage not in {"S6", "S7"}
                or (s6_ready if self.stage == "S6" else s7_admitted)
            ),
        }

    def completion_admission(self) -> tuple[bool, tuple[str, ...]]:
        """Return the S6/S7 hold without becoming a second completion owner."""
        predicates = self.completion_predicates()
        if self.stage == "S6" and not predicates["s6_verification_ready"]:
            return False, ("current_verification_conjunction",)
        if self.stage == "S7" and not predicates["s7_answer_admitted"]:
            return False, ("idealpath_s7_answer",)
        return True, ()

    def backtrack(self, *, target: str, reason_code: str) -> dict[str, object]:
        if (
            target not in STAGES
            or target not in BACKTRACK_TARGETS.get(self.stage, ())
            or reason_code not in REASON_CODES
        ):
            return self._reply(
                False, code="BACKTRACK_INVALID", missing=("typed_backtrack_target",)
            )
        previous = self.stage
        self.stage = target
        self.epoch += 1
        self.transitions.append({
            "from": previous,
            "to": target,
            "state_epoch": self.epoch,
            "kind": "backtrack",
            "reason_code": reason_code,
        })
        if target in {"S2", "S3", "S4", "S5"}:
            self.verification = None
        return self._reply(True, code="BACKTRACK_ADMITTED", status="ADMITTED")

    def host_force_stage_exit(self, *, runtime: object | None = None) -> dict[str, object]:
        """Apply the host fallback once the current stage budget is exhausted."""
        if self.forced_exit_action() is None:
            return self._reply(False, code="HOST_BLOCKED", missing=("stage_exit_budget",))
        if self.stage == "S3":
            from kernel_s3_force import enabled
            if not enabled() or runtime is None:
                return self._reply(False, code="HOST_BLOCKED", missing=("s3force_runtime",))
            if isinstance(self.s3_force_settlement, Mapping):
                return self._reply(
                    False, code="HOST_CAPABILITY_GAP", missing=("host_execution_capability",),
                    status="INCOMPLETE",
                ) | {
                    "fallback": True,
                    "settled": True,
                    "s3_force_result": copy.deepcopy(dict(self.s3_force_settlement)),
                }
            if self.action_count >= self.max_actions:
                return self._reply(False, code="HOST_BLOCKED", missing=("budget",))
            self.action_count += 1
            try:
                from kernel_s3_force import public_spec, stage_answer
                forced = runtime._idealpath_execute_s3_force()
            except (AttributeError, TypeError, ValueError, RuntimeError) as error:
                from kernel_s3_force import capability_gap
                gap = capability_gap(
                    f"s3-force:{self.stage}:{self.epoch}",
                    reason=f"host S3 execution boundary unavailable: {type(error).__name__}",
                    missing=("execution_boundary",),
                )
                self.s3_force_settlement = copy.deepcopy(gap)
                return self._reply(
                    False, code="HOST_CAPABILITY_GAP",
                    missing=("host_execution_capability",), status="INCOMPLETE",
                ) | {"fallback": True, "settled": True,
                    "s3_force_result": gap}
            result = forced if isinstance(forced, Mapping) else {}
            if not result or result.get("code") == "HOST_CAPABILITY_GAP" or result.get("outcome") in {
                    "INCOMPLETE", "ENVIRONMENT", "UNRUNNABLE", "NOT_RUN", "UNKNOWN", "RESOURCE_LIMIT"}:
                if result.get("code") != "HOST_CAPABILITY_GAP":
                    from kernel_s3_force import capability_gap
                    gap = capability_gap(
                        str(result.get("attempt_key") or f"s3-force:{self.stage}:{self.epoch}"),
                        reason="host execution produced no faithful evidence",
                        missing=("execution_evidence",),
                    )
                    gap["underlying_outcome"] = result.get("outcome")
                    gap["underlying_result"] = copy.deepcopy(dict(result))
                    result = gap
                self.s3_force_settlement = copy.deepcopy(dict(result))
                return self._reply(
                    False, code="HOST_CAPABILITY_GAP", missing=("host_execution_capability",),
                    status="INCOMPLETE",
                ) | {
                    "fallback": True,
                    "settled": True,
                    "s3_force_result": copy.deepcopy({
                        key: value for key, value in result.items() if key != "_host_spec"
                    }),
                }
            if result.get("reproduced") is True and result.get("outcome") == "BEHAVIOR_FAIL":
                raw_spec = result.get("_host_spec")
                if isinstance(raw_spec, Mapping):
                    answer = stage_answer(self, raw_spec, result)
                    admission = self.submit({
                        "stage": "S3",
                        "state_epoch": self.epoch,
                        "ticket_hash": self.ticket_hash,
                        "base_revision": self.base_revision,
                        "evidence_refs": [
                            str(item) for item in (
                                result.get("receipt_id"), result.get("check_hash"),
                            ) if item
                        ],
                        "answer": answer,
                    })
                    public = self._reply(
                        bool(admission.get("accepted")),
                        code="S3FORCE_ADMITTED" if admission.get("accepted") else "HOST_BLOCKED",
                        missing=tuple(admission.get("missing_predicates", ())),
                        status="ADMITTED" if admission.get("accepted") else "REJECTED",
                    )
                    public["s3_force_result"] = copy.deepcopy({
                        key: value for key, value in result.items() if key != "_host_spec"
                    })
                    public["s3_force_spec"] = public_spec(raw_spec)
                    public["stage_admission"] = admission
                    self.s3_force_receipts.append(copy.deepcopy(public["s3_force_result"]))
                    self.s3_force_receipts = self.s3_force_receipts[-16:]
                    return public
            return self._reply(False, code="HOST_BLOCKED", missing=("parent_failing_check",)) | {
                "s3_force_result": copy.deepcopy({
                    key: value for key, value in result.items() if key != "_host_spec"
                }),
            }
        if self.stage in {"S4", "S5", "S6", "S7"}:
            return self._stage_exit_exhausted(runtime=runtime)
        return self._admit_host_draft(host_forced=True)

    def dispatch(
            self, action: str, arguments: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Dispatch only actions in the current host-generated menu."""
        args = arguments if isinstance(arguments, Mapping) else {}
        if action not in self._actions():
            return self._reply(
                False, code="CONTRACT_REJECTED", missing=(f"action:{action}",)
            )
        if action == TOOL_NAME:
            return self.submit(args)
        if action == S1_COMPLETION_TOOL_NAME:
            answer = args.get("answer")
            if not isinstance(answer, Mapping):
                answer = {
                    key: args.get(key)
                    for key in ("rows", "entry_hint", "assumptions", "proposal")
                    if key in args
                }
            return self.submit({
                "stage": args.get("stage"),
                "state_epoch": args.get("state_epoch"),
                "ticket_hash": args.get("ticket_hash"),
                "base_revision": args.get("base_revision"),
                "evidence_refs": args.get("evidence_refs", []),
                "answer": answer,
            })
        if action == S2_FORCE_ALTERNATIVE_ACTION:
            from kernel_s2_force import build_receipt, build_s2_localization_draft
            answer = args.get("answer")
            result = self.submit(args)
            if result.get("accepted") is True and isinstance(answer, Mapping):
                draft = build_s2_localization_draft(
                    causal_frontier=self.s2_force_frontier,
                    model_evidence=self.stage_evidence.get("S2", ()),
                    row_ids=tuple(self.rows),
                    target_path=self.s2_force_target.get("path", ""),
                    target_symbol=self.s2_force_target.get("symbol", ""),
                    ticket_hash=self.ticket_hash,
                    base_revision=self.base_revision,
                )
                draft["answer"] = copy.deepcopy(dict(answer))
                draft["provenance"] = "MODEL_TYPED_S2_LOCALIZATION_ALTERNATIVE"
                draft["draft_digest"] = _digest(draft)
                receipt = build_receipt(
                    draft,
                    stage_epoch=max(1, self.epoch - 1),
                    ticket_hash=self.ticket_hash,
                    base_revision=self.base_revision,
                    forced=False,
                    override=True,
                )
                self.s2_force_receipts.append(copy.deepcopy(receipt))
                self.s2_force_receipts = self.s2_force_receipts[-16:]
                result["code"] = "S2FORCE_OVERRIDE"
                result["s2_force_receipt"] = receipt
            return result
        if action == ABSTAIN_ACTION:
            return self._admit_host_draft(args)
        if action == "backtrack":
            return self.backtrack(
                target=str(args.get("target")),
                reason_code=str(args.get("reason_code")),
            )
        if action in {"inspect", "read"}:
            if self.inspect_handler is not None:
                try:
                    result = self.inspect_handler(args)
                except Exception as exc:  # Host boundary returns a typed receipt.
                    return self._reply(
                        False,
                        code="INSPECT_HANDLER_FAILED",
                        missing=(type(exc).__name__,),
                    )
                reply = self._reply(True, code="INSPECT_DISPATCHED", status="ADMITTED")
                reply["inspect_result"] = copy.deepcopy(result)
                return reply
            return self._reply(
                False, code="HOST_BLOCKED", missing=("registered_inspect_handler",)
            )
        if action == LOCALIZE_TOOL_NAME:
            return self._reply(
                False, code="HOST_BLOCKED", missing=("registered_localize_handler",)
            )
        outcome = args.get("outcome")
        if outcome not in {"VERIFIED_FIXED", "NOT_REPRODUCED", "INCOMPLETE"}:
            return self._reply(False, code="INVALID_ANSWER", missing=("typed_outcome",))
        return self.submit({
            "stage": "S7",
            "state_epoch": self.epoch,
            "ticket_hash": self.ticket_hash,
            "base_revision": self.base_revision,
            "evidence_refs": args.get("evidence_refs", []),
            "answer": args,
        })

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "stage": self.stage,
            "state_epoch": self.epoch,
            "ticket_hash": self.ticket_hash,
            "base_revision": self.base_revision,
            "rows": list(self.rows.values()),
            "receipts": list(self.receipts.values()),
            "localization_receipts": list(self.localization_receipts.values()),
            "transitions": list(self.transitions),
            "admitted_failures": dict(self.admitted_failures),
            "frozen_repros": copy.deepcopy(self.frozen_repros),
            "typed_repro_attempts": dict(self.typed_repro_attempts),
            "requirement_table": copy.deepcopy(self.requirement_table),
            "requirement_table_receipt": copy.deepcopy(self.requirement_table_receipt),
            "requirement_ledger": copy.deepcopy(self.requirement_ledger),
            "semantic_review_receipt": copy.deepcopy(self.semantic_review_receipt),
            "admitted_parent_passes": dict(self.admitted_parent_passes),
            "authored_witnesses": dict(self.authored_witnesses),
            "s3_force_attempts": copy.deepcopy(self.s3_force_attempts),
            "s3_force_row_cursor": self.s3_force_row_cursor,
            "s3_force_settlement": copy.deepcopy(self.s3_force_settlement),
            "spine_stage_records": {
                stage: [{"id": ident, "version": version} for ident, version in sorted(refs)]
                for stage, refs in self.spine_stage_records.items()
            },
            "verification": copy.deepcopy(self.verification),
            "answers": copy.deepcopy(self.answers),
            "ticket_focus": copy.deepcopy(self.ticket_focus),
            "action_count": self.action_count,
            "stage_exit_budget": self.stage_exit_budget,
            "stage_turns": dict(self.stage_turns),
            "late_stage_accounting": {
                "settled_requests": sorted(self.late_stage_accounting.settled_requests),
                "settled_actions": sorted(self.late_stage_accounting.settled_actions),
                "receipts": sorted(self.late_stage_accounting.receipts),
                "pending": self.late_stage_accounting.pending,
                "productive": self.late_stage_accounting.productive,
            },
            "host_draft": copy.deepcopy(self.host_draft),
            "navigation_evidence": copy.deepcopy(self.navigation_evidence),
            "stage_evidence": copy.deepcopy(self.stage_evidence),
            "abstentions": copy.deepcopy(self.abstentions),
            "s2_force_frontier": copy.deepcopy(self.s2_force_frontier),
            "s2_force_target": copy.deepcopy(self.s2_force_target),
            "s2_force_receipts": copy.deepcopy(self.s2_force_receipts),
            "s3_force_receipts": copy.deepcopy(self.s3_force_receipts),
            "refusal_tool": self._refusal_tool,
            "refusal_streak": self._refusal_streak,
            "s2_force_menu_notice": copy.deepcopy(self._s2_force_menu_notice),
            "s3_force_menu_notice": copy.deepcopy(self._s3_force_menu_notice),
            "framebuild_receipt": self.framebuild_receipt(),
            "capability_snapshot": (
                self._capability_snapshot.to_dict()
                if self._capability_snapshot is not None else None
            ),
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore only reducer-owned state from a trusted journal snapshot."""
        if snapshot.get("schema") != SCHEMA:
            raise ValueError("invalid IDEALPATH snapshot schema")
        if snapshot.get("ticket_hash") != self.ticket_hash or snapshot.get("base_revision") != self.base_revision:
            raise ValueError("IDEALPATH snapshot identity mismatch")
        stage = str(snapshot.get("stage", ""))
        epoch = snapshot.get("state_epoch")
        if stage not in STAGES or type(epoch) is not int or epoch < 1:
            raise ValueError("invalid IDEALPATH snapshot stage")
        raw_rows = snapshot.get("rows", ())
        raw_receipts = snapshot.get("receipts", ())
        self.stage = stage
        self.epoch = epoch
        self.rows = {
            str(row["row_id"]): dict(copy.deepcopy(dict(row)))
            for row in _items(raw_rows)
            if isinstance(row, Mapping) and _nonempty_text(row.get("row_id"))
        }
        self.receipts = {
            str(receipt["receipt_id"]): dict(copy.deepcopy(dict(receipt)))
            for receipt in _items(raw_receipts)
            if isinstance(receipt, Mapping) and _nonempty_text(receipt.get("receipt_id"))
        }
        localization_receipts = snapshot.get("localization_receipts", ())
        self.localization_receipts = {
            str(receipt["receipt_id"]): dict(copy.deepcopy(dict(receipt)))
            for receipt in _items(localization_receipts)
            if isinstance(receipt, Mapping) and _nonempty_text(receipt.get("receipt_id"))
        }
        self.transitions = [dict(copy.deepcopy(dict(item))) for item in _items(snapshot.get("transitions"))
                            if isinstance(item, Mapping)]
        admitted = snapshot.get("admitted_failures")
        self.admitted_failures = {
            str(key): str(value) for key, value in admitted.items()
        } if isinstance(admitted, Mapping) else {}
        frozen = snapshot.get("frozen_repros")
        self.frozen_repros = {
            str(key): dict(copy.deepcopy(dict(value)))
            for key, value in frozen.items()
            if isinstance(value, Mapping)
        } if isinstance(frozen, Mapping) else {}
        attempts = snapshot.get("typed_repro_attempts")
        self.typed_repro_attempts = {
            str(key): int(value)
            for key, value in attempts.items()
            if type(value) is int and value >= 0
        } if isinstance(attempts, Mapping) else {}
        answers = snapshot.get("answers")
        self.answers = {
            str(key): dict(copy.deepcopy(dict(value))) for key, value in answers.items()
            if isinstance(value, Mapping)
        } if isinstance(answers, Mapping) else {}
        self.ticket_focus = copy.deepcopy(snapshot.get("ticket_focus"))
        verification = snapshot.get("verification")
        self.verification = copy.deepcopy(dict(verification)) if isinstance(verification, Mapping) else None
        requirement_table = snapshot.get("requirement_table")
        if isinstance(requirement_table, Mapping):
            self.bind_requirement_table(requirement_table)
        else:
            self.requirement_table = None
            self._spine_row_refs = set()
            self.spine_stage_records = {}
        requirement_ledger = snapshot.get("requirement_ledger")
        self.requirement_ledger = copy.deepcopy(dict(requirement_ledger)) if isinstance(requirement_ledger, Mapping) else self.requirement_ledger
        requirement_table_receipt = snapshot.get("requirement_table_receipt")
        self.requirement_table_receipt = (
            copy.deepcopy(dict(requirement_table_receipt))
            if isinstance(requirement_table_receipt, Mapping) else None
        )
        semantic_review = snapshot.get("semantic_review_receipt")
        self.semantic_review_receipt = copy.deepcopy(dict(semantic_review)) if isinstance(semantic_review, Mapping) else None
        parent_passes = snapshot.get("admitted_parent_passes")
        self.admitted_parent_passes = {
            str(key): str(value) for key, value in parent_passes.items()
        } if isinstance(parent_passes, Mapping) else {}
        witnesses = snapshot.get("authored_witnesses")
        self.authored_witnesses = {
            str(key): str(value) for key, value in witnesses.items()
        } if isinstance(witnesses, Mapping) else {}
        attempts = snapshot.get("s3_force_attempts")
        self.s3_force_attempts = {
            str(key): dict(copy.deepcopy(dict(value)))
            for key, value in attempts.items() if isinstance(value, Mapping)
        } if isinstance(attempts, Mapping) else {}
        cursor = snapshot.get("s3_force_row_cursor", 0)
        self.s3_force_row_cursor = cursor if type(cursor) is int and cursor >= 0 else 0
        settlement = snapshot.get("s3_force_settlement")
        self.s3_force_settlement = copy.deepcopy(dict(settlement)) if isinstance(settlement, Mapping) else None
        raw_stage_records = snapshot.get("spine_stage_records")
        if isinstance(raw_stage_records, Mapping):
            self.spine_stage_records = {
                str(stage): {
                    (str(item.get("id")), int(item.get("version")))
                    for item in _items(rows)
                    if isinstance(item, Mapping) and _nonempty_text(item.get("id"))
                    and type(item.get("version")) is int and item.get("version") >= 1
                }
                for stage, rows in raw_stage_records.items()
                if str(stage) in STAGES[1:]
            }
        action_count = snapshot.get("action_count", 0)
        self.action_count = action_count if type(action_count) is int and action_count >= 0 else 0
        snapshot_budget = snapshot.get("stage_exit_budget")
        if type(snapshot_budget) is int and snapshot_budget >= 1:
            self.stage_exit_budget = snapshot_budget
            self.late_stage_accounting.capacity = snapshot_budget
            self.late_stage_accounting.pending_limit = snapshot_budget
        stage_turns = snapshot.get("stage_turns")
        if isinstance(stage_turns, Mapping):
            self.stage_turns = {
                stage_name: int(value)
                for stage_name, value in stage_turns.items()
                if str(stage_name) in STAGES and type(value) is int and value >= 0
            }
            self.stage_turns = {
                stage_name: self.stage_turns.get(stage_name, 0) for stage_name in STAGES
            }
        accounting = snapshot.get("late_stage_accounting")
        if isinstance(accounting, Mapping):
            self.late_stage_accounting.settled_requests = {
                str(item) for item in _items(accounting.get("settled_requests")) if str(item)
            }
            self.late_stage_accounting.settled_actions = {
                str(item) for item in _items(accounting.get("settled_actions")) if str(item)
            }
            self.late_stage_accounting.receipts = {
                str(item) for item in _items(accounting.get("receipts")) if str(item)
            }
            pending = accounting.get("pending", 0)
            productive = accounting.get("productive", 0)
            self.late_stage_accounting.pending = min(
                self.late_stage_accounting.pending_limit,
                pending if type(pending) is int and pending >= 0 else 0,
            )
            self.late_stage_accounting.productive = productive if type(productive) is int and productive >= 0 else 0
            self.pending_stage_capacity = self.late_stage_accounting.pending
        draft = snapshot.get("host_draft")
        self.host_draft = copy.deepcopy(dict(draft)) if isinstance(draft, Mapping) else self.host_draft
        navigation = snapshot.get("navigation_evidence")
        self.navigation_evidence = [
            dict(copy.deepcopy(dict(item)))
            for item in _items(navigation)
            if isinstance(item, Mapping)
        ][-64:]
        stage_evidence = snapshot.get("stage_evidence")
        self.stage_evidence = {
            str(stage_name): [
                dict(copy.deepcopy(dict(item)))
                for item in _items(items)
                if isinstance(item, Mapping)
            ][-64:]
            for stage_name, items in stage_evidence.items()
            if str(stage_name) in STAGES and isinstance(items, Sequence)
        } if isinstance(stage_evidence, Mapping) else {}
        abstentions = snapshot.get("abstentions")
        self.abstentions = [
            dict(copy.deepcopy(dict(item)))
            for item in _items(abstentions)
            if isinstance(item, Mapping)
        ][-64:]
        frontier = snapshot.get("s2_force_frontier")
        self.s2_force_frontier = copy.deepcopy(dict(frontier)) if isinstance(frontier, Mapping) else None
        target = snapshot.get("s2_force_target")
        self.s2_force_target = {
            "path": _text(target.get("path", ""), limit=500),
            "symbol": _text(target.get("symbol", ""), limit=300),
        } if isinstance(target, Mapping) else {"path": "", "symbol": ""}
        self.s2_force_receipts = [
            dict(copy.deepcopy(dict(item)))
            for item in _items(snapshot.get("s2_force_receipts"))
            if isinstance(item, Mapping)
        ][-16:]
        self.s3_force_receipts = [
            dict(copy.deepcopy(dict(item)))
            for item in _items(snapshot.get("s3_force_receipts"))
            if isinstance(item, Mapping)
        ][-16:]
        self._refusal_tool = _text(snapshot.get("refusal_tool"), limit=80)
        refusal_streak = snapshot.get("refusal_streak", 0)
        self._refusal_streak = refusal_streak if type(refusal_streak) is int and refusal_streak >= 0 else 0
        notice = snapshot.get("s2_force_menu_notice")
        self._s2_force_menu_notice = copy.deepcopy(dict(notice)) if isinstance(notice, Mapping) else None
        s3_notice = snapshot.get("s3_force_menu_notice")
        self._s3_force_menu_notice = copy.deepcopy(dict(s3_notice)) if isinstance(s3_notice, Mapping) else None
        framebuild_receipt = snapshot.get("framebuild_receipt")
        if isinstance(framebuild_receipt, Mapping):
            self._framebuild_receipt = {
                str(key): copy.deepcopy(value)
                for key, value in framebuild_receipt.items()
                if str(key) not in {"stage", "state_epoch"}
            }
        # A restored owner must reproject its live menu before native work can
        # be admitted; retained receipt state is safe, an old menu is not.
        self._capability_snapshot = None
        self._stale_batch_gate = None
        self._last_stage_turn_marker = None


def attach(
        runtime: object,
        *,
        ticket_hash: str,
        base_revision: str,
        ticket_spans: Sequence[str] = (),
        snapshot: Mapping[str, object] | None = None,
        host_draft: Mapping[str, object] | None = None,
        semantic_reviewer: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        stage_exit_budget: int = DEFAULT_STAGE_EXIT_BUDGET,
) -> IdealPathOwner:
    owner = IdealPathOwner(
        ticket_hash=ticket_hash,
        base_revision=base_revision,
        ticket_spans=ticket_spans,
        host_draft=host_draft,
        semantic_reviewer=semantic_reviewer,
        stage_exit_budget=stage_exit_budget,
    )
    if snapshot is not None:
        owner.restore(snapshot)
    setattr(owner, "_runtime", runtime)
    setattr(runtime, "ideal_path_owner", owner)
    return owner


MODE_ENV = "STATEFRAME_IDEALPATH"


def enabled(runtime: object | None = None, environ: Mapping[str, str] | None = None) -> bool:
    """Resolve the IDEALPATH stage-machine switch (on|off, default on).

    ``off`` means the reducer is never attached: every caller sees
    ``owner(runtime) is None`` and runs the native menu with the ordinary
    completion/honesty gates.  A runtime-declared environment wins over the
    process environment; any other value fails loudly.
    """
    import os
    declared = getattr(runtime, "declared_environment", None)
    if environ is None and isinstance(declared, Mapping) and MODE_ENV in declared:
        environ = {MODE_ENV: str(declared[MODE_ENV])}
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "on")).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value == "on"


def owner(runtime: object) -> IdealPathOwner | None:
    value = getattr(runtime, "ideal_path_owner", None)
    # Release activation can load a module through more than one import
    # boundary; preserve the host-owned object across that boundary by using
    # its reducer surface instead of a fragile class-identity check.
    if isinstance(value, IdealPathOwner):
        return value
    if all(callable(getattr(value, name, None)) for name in ("menu", "dispatch", "snapshot")):
        return value  # type: ignore[return-value]
    return None


__all__ = [
    "ABSTAIN_ACTION", "BACKTRACK_TARGETS", "CAUSAL_LOCALIZE_TOOL_NAME",
    "CERTIFICATION_STATES", "CONTRACT_SCHEMA",
    "DEFAULT_STAGE_EXIT_BUDGET",
    "FAILURE_PHASES", "IdealPathOwner", "OUTCOMES", "RECEIPT_SCHEMA",
    "SCHEMA", "S1_COMPLETION_TOOL_NAME", "STAGES", "STAGE_EXIT_ACTIONS",
    "STAGE_BUDGET_EXIT_ACTIONS",
    "TESTWRITER_TOOL_NAME",
    "StageExitDeadlock", "TOOL_NAME", "assert_all_stage_exit_liveness",
    "assert_stage_exit_liveness", "attach", "enabled", "owner", "requirement_table_tool_schema",
    "stage_exit_liveness", "tool_schema", "backtrack_tool_schema", "stage_tool_schema",
]
