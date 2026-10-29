"""Receipt contract for scheduling and interruption.

Receipts connect observed outcomes to durable work state so scheduling decisions
can be replayed without treating an unverified claim as completion evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MODULE = "detreason.receipt_contract"
VERSION = "1"

# Schema version this contract validates. Must match procedure.SCHEMA_VERSION;
# a mismatch means the emitter moved and this contract is stale — the
# conformance test in test_receipt_contract.py fails loudly on that, on
# purpose, rather than silently accepting drift.
PROCEDURE_RECEIPT_SCHEMA_VERSION = "detreason.procedure_receipt/1"


# ======================================================================
# 1. THE REAL EMITTER SHAPE — cited to procedure.py, not to any doc
# ======================================================================

#: ProcedureResult.receipt() top-level keys, in emission order. Transcribed
#: from procedure.py:565-584. This is the WHOLE top-level shape — there is no
#: top-level `hypotheses`, `gates`, `witnesses`, `abstentions`, or `relations`
#: key. Any consumer expecting those at the top level (as the pre-BURN_LIST
#: `explanation-composer` brief #14 did) is reading a receipt that will never
#: exist; see the module docstring's #223 discussion.
PROCEDURE_RECEIPT_TOP_LEVEL_KEYS: Tuple[str, ...] = (
    "schema_version", "module", "version", "policy_version", "run_id",
    "task_id", "task_kind", "request_digest", "surfaces", "declared_scope",
    "obligation_menu", "attempts", "route", "terminal_phase", "complete",
    "open_items", "escalation",
)

#: Attempt.to_dict() keys (procedure.py:528-544). `hypotheses` and `gates`
#: live HERE, nested inside each attempts[i] entry — never at the receipt's
#: top level.
ATTEMPT_KEYS: Tuple[str, ...] = (
    "attempt", "phases_run", "phase_reached", "hypotheses", "assumptions",
    "checks", "observations", "uncovered_families", "menu_command", "gates",
    "action", "review_questions", "findings", "route", "reasons",
)

#: Hypothesis.to_dict() keys (procedure.py:196-199) — what one entry in an
#: attempt's `hypotheses` list looks like.
HYPOTHESIS_KEYS: Tuple[str, ...] = (
    "id", "family", "claim", "predicted_observation", "check_ids", "status",
    "source", "evidence",
)

#: GateVerdict.to_dict() keys (procedure.py:347-349) — what one entry in an
#: attempt's `gates` list looks like.
GATE_KEYS: Tuple[str, ...] = ("gate", "passed", "reasons")

#: Observation.to_dict() keys (procedure.py:262-265).
OBSERVATION_KEYS: Tuple[str, ...] = (
    "check_id", "value", "evidence_refs", "detail", "attempt",
    "input_digest", "output_digest", "phase",
)

#: escalation_packet() keys (procedure.py:724-736) — the shape of the
#: top-level `escalation` field when it is not None.
ESCALATION_KEYS: Tuple[str, ...] = (
    "task_id", "attempt", "phase", "reasons", "evidence", "alternatives",
    "required_human_input",
)

#: Fields the four design docs each ASSUMED would exist at the receipt's top
#: level, and where each one actually lives in the real emitter. Kept here so
#: a future reader who goes looking for e.g. `witnesses` finds the answer in
#: code, not just in a prose comment.
DOC_ASSUMPTION_TO_REAL_LOCATION: Dict[str, str] = {
    "hypotheses": "attempts[i].hypotheses (Hypothesis.to_dict())",
    "gates": "attempts[i].gates (GateVerdict.to_dict())",
    "witnesses": "does not exist; closest analog is attempts[i].observations "
                 "(Observation.to_dict()) plus attempts[i].route "
                 "(GateVerdict pass/fail per phase)",
    "abstentions": "does not exist as a field; the ABSTAIN route "
                   "(receipt['route'] == procedure.ABSTAIN) plus "
                   "receipt['open_items'] is the closest analog",
    "relations": "does not exist in procedure.py's receipt; a separate "
                 "detreason/relations.py module exists but is not wired into "
                 "ProcedureResult.receipt() as of this change",
}


def get_attempt(receipt: Mapping[str, Any], index: int = -1) -> Dict[str, Any]:
    """Return attempts[index] (default: the latest attempt). Raises IndexError
    on an empty attempts list — never silently returns {}, which would let a
    caller mistake "no attempts happened" for "the attempt had no fields"."""
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        raise TypeError("receipt['attempts'] must be a list")
    if not attempts:
        raise IndexError("receipt has no attempts")
    return attempts[index]


def hypotheses_for_attempt(receipt: Mapping[str, Any],
                           index: int = -1) -> List[Dict[str, Any]]:
    """attempts[index].hypotheses — the ONLY place hypotheses live in a real
    procedure receipt (never at the top level; see DOC_ASSUMPTION_TO_REAL_LOCATION)."""
    return list(get_attempt(receipt, index).get("hypotheses") or [])


def gates_for_attempt(receipt: Mapping[str, Any],
                      index: int = -1) -> List[Dict[str, Any]]:
    """attempts[index].gates — the ONLY place gate verdicts live."""
    return list(get_attempt(receipt, index).get("gates") or [])


def observations_for_attempt(receipt: Mapping[str, Any],
                             index: int = -1) -> List[Dict[str, Any]]:
    return list(get_attempt(receipt, index).get("observations") or [])


def findings_for_attempt(receipt: Mapping[str, Any],
                         index: int = -1) -> List[Dict[str, Any]]:
    return list(get_attempt(receipt, index).get("findings") or [])


def latest_attempt_number(receipt: Mapping[str, Any]) -> int:
    return get_attempt(receipt, -1).get("attempt", 0)


def is_complete(receipt: Mapping[str, Any]) -> bool:
    """True only if the receipt's own `complete` flag says so — never inferred
    from `route` independently, since `complete` is the field procedure.py
    computes FROM route and that's the contract a reader should trust."""
    return bool(receipt.get("complete"))


def blocking_findings(receipt: Mapping[str, Any],
                      index: int = -1) -> List[Dict[str, Any]]:
    """findings in the given attempt (default latest) where `blocking` is
    True. `blocking` is itself a computed property on Finding
    (severity==BLOCKING and value!=TRUE) already baked into to_dict()."""
    return [f for f in findings_for_attempt(receipt, index) if f.get("blocking")]


def escalation_reasons(receipt: Mapping[str, Any]) -> List[str]:
    """The top-level `escalation.reasons`, or [] if the receipt never
    escalated (escalation is None on any non-ESCALATE route)."""
    esc = receipt.get("escalation")
    if not esc:
        return []
    return list(esc.get("reasons") or [])


def validate_procedure_receipt(receipt: Mapping[str, Any]) -> List[str]:
    """Structural conformance check against the REAL emitter shape. Returns a
    list of violations (empty == conforms). Never raises — a malformed
    receipt is data to report, not a crash.

    This is intentionally the single source of truth the conformance test in
    test_receipt_contract.py runs against a REAL ProcedureResult.receipt()
    output, so contract drift between this file and procedure.py fails a test
    instead of silently rotting (rule #223's whole point)."""
    problems: List[str] = []
    if not isinstance(receipt, Mapping):
        return [f"receipt is not a mapping: {type(receipt).__name__}"]

    missing = [k for k in PROCEDURE_RECEIPT_TOP_LEVEL_KEYS if k not in receipt]
    if missing:
        problems.append(f"missing top-level keys: {missing}")
    extra = [k for k in receipt if k not in PROCEDURE_RECEIPT_TOP_LEVEL_KEYS]
    if extra:
        problems.append(f"unexpected top-level keys: {extra}")

    if receipt.get("schema_version") != PROCEDURE_RECEIPT_SCHEMA_VERSION:
        problems.append(
            f"schema_version mismatch: {receipt.get('schema_version')!r} "
            f"!= {PROCEDURE_RECEIPT_SCHEMA_VERSION!r}")

    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        problems.append("attempts is not a list")
    else:
        for i, a in enumerate(attempts):
            if not isinstance(a, Mapping):
                problems.append(f"attempts[{i}] is not a mapping")
                continue
            a_missing = [k for k in ATTEMPT_KEYS if k not in a]
            if a_missing:
                problems.append(f"attempts[{i}] missing keys: {a_missing}")
            a_extra = [k for k in a if k not in ATTEMPT_KEYS]
            if a_extra:
                problems.append(f"attempts[{i}] unexpected keys: {a_extra}")
            for j, h in enumerate(a.get("hypotheses") or []):
                h_missing = [k for k in HYPOTHESIS_KEYS if k not in h]
                if h_missing:
                    problems.append(
                        f"attempts[{i}].hypotheses[{j}] missing keys: {h_missing}")
            for j, g in enumerate(a.get("gates") or []):
                g_missing = [k for k in GATE_KEYS if k not in g]
                if g_missing:
                    problems.append(
                        f"attempts[{i}].gates[{j}] missing keys: {g_missing}")

    esc = receipt.get("escalation")
    if esc is not None:
        if not isinstance(esc, Mapping):
            problems.append("escalation is neither null nor a mapping")
        else:
            e_missing = [k for k in ESCALATION_KEYS if k not in esc]
            if e_missing:
                problems.append(f"escalation missing keys: {e_missing}")

    # top-level fields a stray doc-shaped reader might reach for, that must
    # NEVER appear at this level (they'd mean the emitter changed shape
    # without this contract being updated to match).
    for ghost in ("hypotheses", "gates", "witnesses", "abstentions", "relations"):
        if ghost in receipt:
            problems.append(
                f"top-level {ghost!r} present — the real emitter never puts "
                f"this at the top level (see DOC_ASSUMPTION_TO_REAL_LOCATION)")

    return problems


# ======================================================================
# 2. THE RECONCILED HANDOFF ENVELOPE — design intent, not yet an emitter
# ======================================================================
#
# None of the four design-doc shapes below have a running emitter. This
# section canonizes ONE reconciled shape so that when B16/B17/the kernel
# finally write one, they build to a single target instead of re-deriving
# four incompatible ones. Every field is cited to its source doc.

#: mode_selector.HandoffContract (§5.3) verbatim — untouched, because
#: juggler_interrupts is explicit that scheduling receipts EXTEND the
#: procedure receipt, and mode_longhorizon is explicit that
#: ModeTransition.handoff_hash should HASH this contract, not replace it.
#: This is the payload every consumer that wants "what was handed off" reads.
HANDOFF_CONTRACT_KEYS: Tuple[str, ...] = (
    "source_mode", "target_mode", "input_artifacts", "verified_properties",
    "unresolved_questions", "evidence_refs", "prohibited_assumptions",
    "target_success_conditions",
)

#: juggler_interrupts.LaneCheckpointReceipt (the doc's own words: "A receipt
#: must add scheduling fields to the EXISTING procedure evidence" — so this is
#: additive wrapping, never a replacement shape). `procedure_receipt_digest`
#: is this module's reconciliation of that instruction: a checkpoint always
#: carries a digest of the real procedure.py receipt it wraps, never a copy
#: of its fields.
LANE_CHECKPOINT_KEYS: Tuple[str, ...] = (
    "lane_id", "predecessor_receipt", "consumed_event_cursor",
    "deferred_event_cursor", "state_digest_before", "state_digest_after",
    "rehydration_digest", "model_output_digest", "next_action",
    "open_obligations", "preempted_by_event_id", "world_digest", "status",
    "procedure_receipt_digest",   # RECONCILED addition, see docstring above
)

#: mode_longhorizon.ModeTransition (§2) verbatim, EXCEPT `handoff_hash`'s
#: meaning is now pinned: it is `digest(HandoffContract)`, computed by
#: `handoff_hash()` below — never a second, independently-shaped payload.
MODE_TRANSITION_KEYS: Tuple[str, ...] = (
    "transition_id", "project_id", "from_goal_id", "from_mode", "to_goal_id",
    "to_mode", "source_exit_receipt_id", "input_snapshot_hash",
    "output_artifact_hash", "contract_set_hash", "constraint_set_hash",
    "handoff_hash", "entry_gate_ids", "unresolved_unknowns",
    "invalidated_descendants", "status",
)

#: mode_selector.Receipt (§9) verbatim — the kernel-level append-only
#: transition receipt. Broader than one procedure.py receipt; one PROCEDURE
#: receipt (identified by `run_id`) is expected to appear inside this
#: receipt's `evidence_refs` / `procedure_steps`, never be duplicated by it.
KERNEL_RECEIPT_KEYS: Tuple[str, ...] = (
    "receipt_id", "parent_receipt_id", "task_id", "contract_hash", "mode",
    "stack_position", "lane_id", "candidates_considered", "selected_candidate",
    "procedure_steps", "evidence_refs", "evidence_snapshots", "gate_results",
    "witness_results", "side_effects", "authority_refs", "abstentions",
    "output_refs", "output_hashes", "checker_versions",
    "model_and_tool_versions", "deterministic_ordering_key",
)


def _canonical(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # pragma: no cover - default=str makes this near-impossible
        return str(obj)


def digest(obj: Any) -> str:
    """Same content-addressing shape as procedure.digest() (sha256, first-12
    hex, "sha256:" prefixed) so a handoff_hash and a procedure receipt's
    run_id/request_digest are directly comparable-looking artifacts."""
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class HandoffContract:
    """mode_selector.HandoffContract (§5.3), field-for-field."""

    source_mode: str
    target_mode: str
    input_artifacts: Tuple[str, ...] = ()
    verified_properties: Tuple[str, ...] = ()
    unresolved_questions: Tuple[str, ...] = ()
    evidence_refs: Tuple[str, ...] = ()
    prohibited_assumptions: Tuple[str, ...] = ()
    target_success_conditions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_mode:
            raise ValueError("source_mode is required")
        if not self.target_mode:
            raise ValueError("target_mode is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "target_mode": self.target_mode,
            "input_artifacts": list(self.input_artifacts),
            "verified_properties": list(self.verified_properties),
            "unresolved_questions": list(self.unresolved_questions),
            "evidence_refs": list(self.evidence_refs),
            "prohibited_assumptions": list(self.prohibited_assumptions),
            "target_success_conditions": list(self.target_success_conditions),
        }


def handoff_hash(contract: HandoffContract) -> str:
    """The RECONCILED meaning of ModeTransition.handoff_hash: a digest of the
    HandoffContract payload, never a freestanding field. Callers building a
    ModeTransition MUST derive handoff_hash this way rather than inventing
    their own hash of ad hoc fields — that is the whole point of W0-3's
    reconciliation instruction ("ModeTransition.handoff_hash should hash a
    HandoffContract, not replace it")."""
    return digest(contract.to_dict())


@dataclass(frozen=True)
class LaneCheckpoint:
    """juggler_interrupts.LaneCheckpointReceipt, plus the RECONCILED
    `procedure_receipt_digest` field wiring it to a real procedure.py
    receipt rather than re-deriving procedure state independently."""

    lane_id: str
    predecessor_receipt: Optional[str] = None
    consumed_event_cursor: int = 0
    deferred_event_cursor: int = 0
    state_digest_before: str = ""
    state_digest_after: str = ""
    rehydration_digest: str = ""
    model_output_digest: str = ""
    next_action: str = ""
    open_obligations: Tuple[str, ...] = ()
    preempted_by_event_id: Optional[str] = None
    world_digest: str = ""
    status: str = ""
    procedure_receipt_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "predecessor_receipt": self.predecessor_receipt,
            "consumed_event_cursor": self.consumed_event_cursor,
            "deferred_event_cursor": self.deferred_event_cursor,
            "state_digest_before": self.state_digest_before,
            "state_digest_after": self.state_digest_after,
            "rehydration_digest": self.rehydration_digest,
            "model_output_digest": self.model_output_digest,
            "next_action": self.next_action,
            "open_obligations": list(self.open_obligations),
            "preempted_by_event_id": self.preempted_by_event_id,
            "world_digest": self.world_digest,
            "status": self.status,
            "procedure_receipt_digest": self.procedure_receipt_digest,
        }


def checkpoint_from_procedure_receipt(
    *, lane_id: str, procedure_receipt: Mapping[str, Any],
    predecessor_receipt: Optional[str] = None,
    consumed_event_cursor: int = 0, deferred_event_cursor: int = 0,
    **overrides: Any,
) -> LaneCheckpoint:
    """The ONE sanctioned way to build a LaneCheckpoint from a real
    procedure.py receipt: hash it in, never copy its fields out. This is the
    structural enforcement of juggler_interrupts's "add scheduling fields to
    the EXISTING procedure evidence" instruction — a checkpoint that reaches
    into `procedure_receipt['attempts']` to duplicate data is doing it wrong.
    """
    problems = validate_procedure_receipt(procedure_receipt)
    if problems:
        raise ValueError(
            f"refusing to checkpoint a non-conformant procedure receipt: {problems}")
    return LaneCheckpoint(
        lane_id=lane_id,
        predecessor_receipt=predecessor_receipt,
        consumed_event_cursor=consumed_event_cursor,
        deferred_event_cursor=deferred_event_cursor,
        status=procedure_receipt.get("route", ""),
        procedure_receipt_digest=digest(procedure_receipt),
        **overrides,
    )


@dataclass(frozen=True)
class ModeTransition:
    """mode_longhorizon.ModeTransition (§2), field-for-field, with
    handoff_hash's meaning pinned to `handoff_hash(contract)` above."""

    transition_id: str
    project_id: str
    from_goal_id: Optional[str]
    from_mode: Optional[str]
    to_goal_id: str
    to_mode: str
    source_exit_receipt_id: Optional[str] = None
    input_snapshot_hash: str = ""
    output_artifact_hash: str = ""
    contract_set_hash: str = ""
    constraint_set_hash: str = ""
    handoff_hash: str = ""
    entry_gate_ids: Tuple[str, ...] = ()
    unresolved_unknowns: Tuple[str, ...] = ()
    invalidated_descendants: Tuple[str, ...] = ()
    status: str = ""

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id is required")
        if not self.to_mode:
            raise ValueError("to_mode is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "project_id": self.project_id,
            "from_goal_id": self.from_goal_id,
            "from_mode": self.from_mode,
            "to_goal_id": self.to_goal_id,
            "to_mode": self.to_mode,
            "source_exit_receipt_id": self.source_exit_receipt_id,
            "input_snapshot_hash": self.input_snapshot_hash,
            "output_artifact_hash": self.output_artifact_hash,
            "contract_set_hash": self.contract_set_hash,
            "constraint_set_hash": self.constraint_set_hash,
            "handoff_hash": self.handoff_hash,
            "entry_gate_ids": list(self.entry_gate_ids),
            "unresolved_unknowns": list(self.unresolved_unknowns),
            "invalidated_descendants": list(self.invalidated_descendants),
            "status": self.status,
        }


def mode_transition_for_handoff(
    *, transition_id: str, project_id: str, to_goal_id: str, to_mode: str,
    contract: HandoffContract, from_goal_id: Optional[str] = None,
    from_mode: Optional[str] = None, **overrides: Any,
) -> ModeTransition:
    """The ONE sanctioned way to build a ModeTransition around a
    HandoffContract: handoff_hash is ALWAYS derived from the contract via
    `handoff_hash()`, never passed in independently. This is the code-level
    enforcement of W0-3's reconciliation instruction. Any caller-supplied
    ``handoff_hash`` override is deliberately discarded — the derived value
    always wins."""
    overrides.pop("handoff_hash", None)
    return ModeTransition(
        transition_id=transition_id, project_id=project_id,
        from_goal_id=from_goal_id, from_mode=from_mode,
        to_goal_id=to_goal_id, to_mode=to_mode,
        handoff_hash=handoff_hash(contract),
        **overrides,
    )
