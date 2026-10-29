"""The ONE obligation system — an append-only liability register.

WHY THIS FILE EXISTS (BURN_LIST W0-4)
=====================================
Four designs invented the same concept four times:

* `procedure.py:enumerate_obligations(task_kind, surfaces)` — ALREADY SHIPS.
  It is the deterministic coverage contract for a task's obligation families.
* `w60/a_obligation_ledger.md` — invented an `Obligation` type and a full
  lifecycle from scratch and never once referenced the shipping function
  despite the near-identical name.
* `w60/a_park_and_return.md` — independently invented `ReturnObligation` +
  `ParkedItem` with a DIFFERENT state machine.
* `design-pre-mortem-generator.md` — a third, `ProcedureObligation`.

RECONCILIATION (the seam, stated once, in code)
-----------------------------------------------
There is ONE `Obligation` dataclass and ONE state machine. The others become
*kinds* and *producers*:

    procedure.enumerate_obligations  --producer-->  Obligation(kind=PROCEDURE_FAMILY)
    park-and-return ReturnObligation ---kind---->   Obligation(kind=RETURN)
    premortem ProcedureObligation ----kind----->    Obligation(kind=PREMORTEM)

`enumerate_obligations` is IMPORTED, never re-listed here. `families_for_task`
is a thin adapter; `test_obligations.py` asserts byte-equality with procedure's
output so a fork is impossible without a red test (#223: one key, one spelling;
here, one family list, one owner).

ONE PRIORITY FORMULA (the second half of W0-4)
----------------------------------------------
Two competing weighted sums fed one scheduler in the source docs:

    park-and-return:   priority_class + deadline_pressure + bounded_age
                       + fairness_credit - recent_service_cost
    ledger §17:        explicit_priority + age_risk + due_risk
                       + freshness_risk + dependency_criticality
                       + promise_breach_risk

Two weighted sums feeding one scheduler is a brick (#211). PICKED: the
park-and-return SHAPE, because it is the only one with bounded aging and a
fairness credit, i.e. the only one with an anti-starvation property. The
ledger's two distinctive terms are folded in as INPUTS to existing components,
not as a second sum: `promise_breach_risk` feeds deadline_pressure,
`dependency_criticality` feeds priority_class. There is exactly one
`service_score()` and it returns one integer.

HARD RULES ENCODED HERE
-----------------------
* A model may propose; only the kernel discharges. "done" is never evidence.
* Rot never silently discharges. A rotten obligation is still OPEN.
* Closing is a SEPARATE operation from answering. The post-answer gate rejects
  `close` whenever the act creates a material follow-up (FALSE_CLOSE).
* Unknown impact FAILS CLOSED (NEEDS_REVIEW), it never closes.
* Deduplication never resets age (`first_seen_at` is immutable).
* "Still parked" is never a successful terminal state.
* Session completion is a kernel command with no model override.

Integers only, everywhere: all times are INTEGER epoch microseconds supplied by
the caller. There is no ambient clock read in this module — a literal-date or
ambient-clock fixture meeting a relative window arms itself into vacuity
(#214), and a scheduler that reads `time.time()` internally cannot be tested
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from typing import Sequence, Tuple

from .procedure import enumerate_obligations, SURFACES, TASK_KINDS
from .snapshot import cse1_digest

__all__ = [
    "ObligationError", "GateError",
    "HUMAN_REQUEST", "AGENT_PROMISE", "PENDING_HUMAN_ANSWER",
    "STANDING_RULE_OCCURRENCE", "FOLLOW_UP", "RETURN", "PREMORTEM",
    "PROCEDURE_FAMILY", "PLANNED_ARTIFACT", "OBLIGATION_KINDS", "NEVER_MERGE_KINDS",
    "PROPOSED", "OPEN", "BLOCKED", "WAITING_FOR_HUMAN", "READY",
    "IN_PROGRESS", "DISCHARGED", "SUPERSEDED", "CANCELLED",
    "STATUSES", "TERMINAL_STATUSES", "OPEN_STATUSES",
    "FRESH", "AT_RISK", "ROTTEN",
    "AnswerContract", "ArtifactContract", "ActionContract",
    "FollowUpDecisionContract", "CONTRACT_ACT_KINDS", "contract_hash",
    "FreshnessPolicy", "Evidence", "Act", "Obligation", "DependencyEdge",
    "SatisfactionCheck", "satisfies", "freshness_state",
    "ClosureAttempt", "ImpactReport", "ClosureResult",
    "DISCHARGED_OUTCOME", "FALSE_CLOSE", "NEEDS_REVIEW", "REJECTED",
    "ObligationLedger", "ParkedItem", "ParkReceipt", "AgingReport",
    "SessionReceipt", "SessionCompletionBlocked", "planned_artifact_obligations",
    "service_score", "families_for_task", "register_procedure_families",
    "P0", "P1", "P2", "P3", "PRIORITY_CLASSES",
    "NEXT_CHECKPOINT", "END_CURRENT_LANE", "SESSION_REVIEW",
    "SERVICE_DEADLINES", "EVENT_NAMES",
]

MICROS_PER_SECOND = 1_000_000


class ObligationError(RuntimeError):
    """A violation of the obligation contract."""


class GateError(ObligationError):
    """The closure gate refused an operation outright."""


# ------------------------------------------------------------------ kinds
HUMAN_REQUEST = "human_request"
AGENT_PROMISE = "agent_promise"
PENDING_HUMAN_ANSWER = "pending_human_answer"
STANDING_RULE_OCCURRENCE = "standing_rule_occurrence"
FOLLOW_UP = "follow_up"
RETURN = "return"                      # <- park-and-return's ReturnObligation
PREMORTEM = "premortem"                # <- premortem's ProcedureObligation
PROCEDURE_FAMILY = "procedure_family"  # <- enumerate_obligations' families
PLANNED_ARTIFACT = "planned_artifact"  # Planned input ledger.

OBLIGATION_KINDS = frozenset({
    HUMAN_REQUEST, AGENT_PROMISE, PENDING_HUMAN_ANSWER,
    STANDING_RULE_OCCURRENCE, FOLLOW_UP, RETURN, PREMORTEM, PROCEDURE_FAMILY,
    PLANNED_ARTIFACT,
})

#: Kinds that must NEVER be semantically merged. Two distinct agent promises
#: are two distinct liabilities even when the words match; merging them lets
#: one delivery discharge two commitments.
NEVER_MERGE_KINDS = frozenset({AGENT_PROMISE, RETURN, PLANNED_ARTIFACT})

# --------------------------------------------------------------- statuses
PROPOSED = "proposed"
OPEN = "open"
BLOCKED = "blocked"
WAITING_FOR_HUMAN = "waiting_for_human"
READY = "ready"
IN_PROGRESS = "in_progress"
DISCHARGED = "discharged"
SUPERSEDED = "superseded"
CANCELLED = "cancelled"

STATUSES = (PROPOSED, OPEN, BLOCKED, WAITING_FOR_HUMAN, READY, IN_PROGRESS,
            DISCHARGED, SUPERSEDED, CANCELLED)
TERMINAL_STATUSES = frozenset({DISCHARGED, SUPERSEDED, CANCELLED})
OPEN_STATUSES = frozenset(STATUSES) - TERMINAL_STATUSES

# Freshness is ORTHOGONAL to status. A rotten obligation is still open.
FRESH = "fresh"
AT_RISK = "at_risk"
ROTTEN = "rotten"

# ------------------------------------------------------------- priorities
P0, P1, P2, P3 = "P0", "P1", "P2", "P3"
PRIORITY_CLASSES = (P0, P1, P2, P3)
_PRIORITY_WEIGHT = {P0: 1000, P1: 600, P2: 250, P3: 60}

NEXT_CHECKPOINT = "next_checkpoint"
END_CURRENT_LANE = "end_current_lane"
SESSION_REVIEW = "session_review"
SERVICE_DEADLINES = (NEXT_CHECKPOINT, END_CURRENT_LANE, SESSION_REVIEW)

EVENT_NAMES = (
    "UTTERANCE_RECEIVED", "ASK_EXTRACTED", "OBLIGATION_REGISTERED",
    "OBLIGATION_DEDUPLICATED", "PROMISE_REGISTERED", "OBLIGATION_CLAIMED",
    "OBLIGATION_BLOCKED", "OBLIGATION_ROTTEN", "RE_DERIVATION_REQUESTED",
    "ACT_RECORDED", "CLOSURE_GATE_EVALUATED", "FALSE_CLOSE_REJECTED",
    "OBLIGATION_DISCHARGED", "FOLLOW_UP_REGISTERED", "OBLIGATION_SUPERSEDED",
    "OBLIGATION_CANCELLED", "PARK_INTENT", "PARK_RECEIPT",
    "PARK_ESCALATED", "PARK_CANCELLED", "PARK_AGING_ALARM",
    "SESSION_PARKED", "SESSION_RESUMED", "SESSION_COMPLETION_BLOCKED",
    "SESSION_COMPLETED", "PLANNED_ARTIFACT_DECLARED",
    "PLANNED_ARTIFACT_FULFILLED", "PLANNED_ARTIFACT_REPLANNED",
    "PLANNED_ARTIFACT_BROKEN",
)


# =========================================================================
# Typed satisfaction contracts — a CLOSED tagged union.
# =========================================================================

@dataclass(frozen=True)
class AnswerContract:
    kind: str = field(default="answer", init=False)
    answer_schema_ref: str = ""
    expected_actor: str = "agent"          # "human" | "agent"
    required_fields: Tuple[str, ...] = ()
    acceptance_predicate_ref: str = ""

    def __post_init__(self):
        if self.expected_actor not in ("human", "agent"):
            raise ObligationError(
                f"answer contract expected_actor must be human|agent, "
                f"got {self.expected_actor!r}")


@dataclass(frozen=True)
class ArtifactContract:
    kind: str = field(default="artifact", init=False)
    artifact_type: str = ""
    recipient: str = ""
    acceptance_predicate_ref: str = ""
    required_version: Optional[str] = None
    delivery_channel: str = ""
    expected_actor: str = "agent"

    def __post_init__(self):
        if not self.artifact_type:
            raise ObligationError("artifact contract needs an artifact_type")


@dataclass(frozen=True)
class ActionContract:
    kind: str = field(default="action", init=False)
    action_type: str = ""
    target_ref: str = ""
    expected_actor: str = "agent"
    preconditions: Tuple[str, ...] = ()
    postconditions: Tuple[str, ...] = ()
    idempotency_key: str = ""
    authorization_scope: str = ""

    def __post_init__(self):
        if not self.action_type:
            raise ObligationError("action contract needs an action_type")
        if not self.postconditions:
            # An action with no postcondition can only ever be satisfied by an
            # agent SAYING it did the thing. That is exactly the failure mode.
            raise ObligationError(
                "action contract needs at least one postcondition; an action "
                "is not satisfied because it was attempted")


@dataclass(frozen=True)
class FollowUpDecisionContract:
    kind: str = field(default="follow_up_decision", init=False)
    decision_ref: str = ""
    decision_maker: str = "human"
    options: Tuple[str, ...] = ()
    selected_option_required: bool = True
    expected_actor: str = "human"

    def __post_init__(self):
        if len(self.options) < 2:
            raise ObligationError(
                "a decision contract needs at least two options")


CONTRACT_ACT_KINDS = {
    "answer": "answer_provided",
    "artifact": "artifact_produced",
    "action": "action_completed",
    "follow_up_decision": "decision_recorded",
}

ACT_KINDS = frozenset(CONTRACT_ACT_KINDS.values())


def _frozen_digest(obj: Any) -> str:
    """Content identity of any frozen dataclass in this module."""
    body: Dict[str, Any] = {"__type__": type(obj).__name__}
    for name, value in sorted(vars(obj).items()):
        body[name] = list(value) if isinstance(value, tuple) else value
    return cse1_digest(body)


def contract_hash(contract: Any) -> str:
    """Content identity of a contract, used for exact and semantic dedup."""
    return _frozen_digest(contract)


# =========================================================================
# Freshness, evidence, acts.
# =========================================================================

_SENSITIVITIES = ("none", "session", "live", "deadline_bound",
                  "source_versioned")


@dataclass(frozen=True)
class FreshnessPolicy:
    sensitivity: str = "none"
    max_age_micros: Optional[int] = None
    source_version: Optional[str] = None
    at_risk_fraction_num: int = 3          # at_risk once age > 3/4 of max_age
    at_risk_fraction_den: int = 4
    stale_behavior: str = "reject_replay"  # or "revalidate_before_use"
    rederive_procedure_ref: Optional[str] = None

    def __post_init__(self):
        if self.sensitivity not in _SENSITIVITIES:
            raise ObligationError(
                f"unknown freshness sensitivity {self.sensitivity!r}; "
                f"known: {_SENSITIVITIES}")
        if self.sensitivity in ("live", "deadline_bound"):
            if self.max_age_micros is None:
                raise ObligationError(
                    f"{self.sensitivity} freshness requires max_age_micros; "
                    f"unknown freshness must fail closed, not default to "
                    f"'forever fresh'")
        if self.sensitivity == "source_versioned" and not self.source_version:
            raise ObligationError(
                "source_versioned freshness requires a source_version")
        if self.max_age_micros is not None and self.max_age_micros <= 0:
            raise ObligationError("max_age_micros must be positive")


@dataclass(frozen=True)
class Evidence:
    ref: str
    observed_at: int                       # integer epoch micros
    source_version: Optional[str] = None
    as_of: Optional[int] = None

    def __post_init__(self):
        if not self.ref:
            raise ObligationError("evidence needs a ref")
        _require_micros(self.observed_at, "evidence.observed_at")


@dataclass(frozen=True)
class Act:
    id: str
    kind: str
    actor: str                             # human | agent | system | external
    occurred_at: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence: Tuple[Evidence, ...] = ()
    procedure_receipt_id: Optional[str] = None
    procedure_status: Optional[str] = None  # completed|partial|failed|cancelled
    postconditions_proven: Tuple[str, ...] = ()
    source_version: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ACT_KINDS:
            raise ObligationError(
                f"unknown act kind {self.kind!r}; known: {sorted(ACT_KINDS)}")
        if self.actor not in ("human", "agent", "system", "external"):
            raise ObligationError(f"unknown actor {self.actor!r}")
        _require_micros(self.occurred_at, "act.occurred_at")


def _require_micros(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObligationError(
            f"{label} must be an INTEGER epoch-microseconds value, got "
            f"{value!r}")
    if value < 0:
        raise ObligationError(f"{label} must be non-negative")


# =========================================================================
# The obligation.
# =========================================================================

@dataclass(frozen=True)
class DependencyEdge:
    kind: str                              # requires|feeds|blocks|spawned_by|
    target_id: str                         # supersedes|invalidates

    def __post_init__(self):
        known = ("requires", "feeds", "blocks", "spawned_by", "supersedes",
                 "invalidates")
        if self.kind not in known:
            raise ObligationError(
                f"unknown dependency kind {self.kind!r}; known: {known}")


@dataclass(frozen=True)
class Obligation:
    id: str
    workspace_id: str
    session_id: str
    kind: str
    title: str
    contract: Any
    origin_event_id: str
    owner: str = "agent"                   # agent|human|system|external
    status: str = OPEN
    priority_class: str = P2
    deadline_policy: str = SESSION_REVIEW
    dependency_criticality: int = 0        # folded into priority (see module doc)
    promise_breach_risk: int = 0           # folded into deadline pressure
    freshness: FreshnessPolicy = field(default_factory=FreshnessPolicy)
    dependencies: Tuple[DependencyEdge, ...] = ()
    normalized_intent: str = ""
    target_ref: str = ""
    dependency_anchor: str = ""
    span_hash: str = ""
    first_seen_at: int = 0
    created_at: int = 0
    last_progress_at: int = 0
    due_at: Optional[int] = None
    registration_count: int = 1
    occurrence_refs: Tuple[str, ...] = ()
    satisfying_act_id: Optional[str] = None
    evidence_observed_at: Optional[int] = None
    evidence_source_version: Optional[str] = None
    cancel_reason: str = ""
    cancel_authorized_by: str = ""
    parent_id: Optional[str] = None
    park_id: Optional[str] = None
    procedure_family: Optional[str] = None
    # Planned-artifact fields are inert for every older obligation kind.  They
    # live on the existing obligation record so admission, aging, and closure
    # cannot drift into a second liability register.
    artifact_path: str = ""
    artifact_module: str = ""
    artifact_exports: Tuple[str, ...] = ()
    artifact_owner_work_id: str = ""
    artifact_requirement_ids: Tuple[str, ...] = ()
    artifact_consumer_refs: Tuple[str, ...] = ()
    artifact_contract_refs: Tuple[str, ...] = ()
    artifact_predicates: Tuple[str, ...] = ()
    artifact_admission_origin: str = ""
    artifact_admission_receipt: str = ""
    artifact_disposition: str = "planned"
    artifact_health: str = "planned"
    artifact_lineage_id: str = ""
    artifact_required_before: str = "feature_acceptance"
    artifact_fulfillment_boundary: str = ""
    artifact_revisit_rule: str = ""
    artifact_review_count: int = 0
    artifact_deletion_break: bool = False
    last_serviced_at: Optional[int] = None
    service_count: int = 0

    def __post_init__(self):
        if self.kind not in OBLIGATION_KINDS:
            raise ObligationError(f"unknown obligation kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ObligationError(f"unknown status {self.status!r}")
        if self.priority_class not in PRIORITY_CLASSES:
            raise ObligationError(
                f"unknown priority class {self.priority_class!r}")
        if self.deadline_policy not in SERVICE_DEADLINES:
            raise ObligationError(
                f"unknown deadline policy {self.deadline_policy!r}")
        if self.contract.kind not in CONTRACT_ACT_KINDS:
            raise ObligationError(
                f"unknown contract kind {self.contract.kind!r}")
        if not self.workspace_id:
            raise ObligationError("workspace_id is required")
        if self.kind == PLANNED_ARTIFACT:
            if not self.artifact_path or not self.artifact_owner_work_id:
                raise ObligationError(
                    "planned_artifact obligations require an exact path and owner_work_id")
            normalized_path = self.artifact_path.replace("\\", "/")
            if (normalized_path.startswith("/")
                    or any(part in {"", ".", ".."}
                           for part in normalized_path.split("/"))
                    or any(char in normalized_path for char in "*?[]")):
                raise ObligationError("planned_artifact path may not contain wildcards")
            if not self.artifact_predicates:
                raise ObligationError("planned_artifact needs fulfillment predicates")

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def is_planned_artifact(self) -> bool:
        return self.kind == PLANNED_ARTIFACT

    @property
    def required_act_kind(self) -> str:
        return CONTRACT_ACT_KINDS[self.contract.kind]

    def age_micros(self, now: int) -> int:
        _require_micros(now, "now")
        return max(0, now - self.first_seen_at)

    # ---- identity keys
    def registration_key(self) -> str:
        """Exact idempotency: the same source event + span + contract is the
        SAME registration, no matter how many times extraction retries."""
        return cse1_digest({
            "workspace_id": self.workspace_id,
            "source_event_id": self.origin_event_id,
            "span_hash": self.span_hash,
            "contract_hash": contract_hash(self.contract),
        })

    def canonical_key(self) -> str:
        """Semantic dedup key. Every field here is a reason two asks are
        DIFFERENT; leaving one out is how two liabilities collapse into one."""
        return cse1_digest({
            "workspace_id": self.workspace_id,
            "kind": self.kind,
            "normalized_intent": self.normalized_intent,
            "contract_hash": contract_hash(self.contract),
            "target_ref": self.target_ref,
            "owner": self.owner,
            "dependency_anchor": self.dependency_anchor,
            "freshness": _frozen_digest(self.freshness),
            "due_at": self.due_at,
        })


# =========================================================================
# Freshness evaluation and the satisfaction predicate.
# =========================================================================

def freshness_state(policy: FreshnessPolicy, *, observed_at: Optional[int],
                    now: int, source_version: Optional[str] = None) -> str:
    """FRESH / AT_RISK / ROTTEN. Unknown freshness fails CLOSED to ROTTEN."""
    _require_micros(now, "now")
    if policy.sensitivity == "none":
        return FRESH
    if observed_at is None:
        return ROTTEN                     # fail closed: unknown is not fresh
    if policy.sensitivity == "source_versioned":
        if source_version is None:
            return ROTTEN
        return FRESH if source_version == policy.source_version else ROTTEN
    if policy.max_age_micros is None:
        # "session" sensitivity with no cap: fresh within the session, and the
        # ledger's resume path re-checks it explicitly.
        return FRESH
    age = now - observed_at
    if age > policy.max_age_micros:
        return ROTTEN
    threshold = (policy.max_age_micros * policy.at_risk_fraction_num
                 // policy.at_risk_fraction_den)
    return AT_RISK if age > threshold else FRESH


@dataclass(frozen=True)
class SatisfactionCheck:
    ok: bool
    failures: Tuple[str, ...] = ()


def satisfies(obligation: Obligation, act: Act, *, now: int,
              dependencies_discharged: bool = True,
              acceptance_predicate: Optional[Callable[[Obligation, Act],
                                                      bool]] = None
              ) -> SatisfactionCheck:
    """The deterministic predicate. The model may interpret; the kernel
    evaluates. Every clause here has its own failure string so a rejection is
    never a mystery."""
    failures: List[str] = []
    contract = obligation.contract

    if obligation.is_terminal:
        failures.append(f"obligation is already {obligation.status}")
    if act.kind != obligation.required_act_kind:
        failures.append(
            f"incompatible act kind {act.kind!r}; contract "
            f"{contract.kind!r} requires {obligation.required_act_kind!r}")

    expected_actor = getattr(contract, "expected_actor", None)
    if expected_actor is not None and act.actor != expected_actor:
        failures.append(
            f"unauthorized actor {act.actor!r}; contract expects "
            f"{expected_actor!r}")
    if (obligation.kind == PENDING_HUMAN_ANSWER and act.actor != "human"):
        failures.append(
            "a pending human answer cannot be satisfied by the agent "
            "answering its own question")

    if contract.kind == "answer":
        missing = [f for f in contract.required_fields
                   if f not in act.payload]
        if missing:
            failures.append(f"answer payload missing fields {missing}")
    elif contract.kind == "artifact":
        if act.payload.get("artifact_type") != contract.artifact_type:
            failures.append(
                f"artifact type {act.payload.get('artifact_type')!r} != "
                f"required {contract.artifact_type!r}")
        if contract.delivery_channel and not act.payload.get("delivered_to"):
            failures.append(
                "artifact requires delivery but was never delivered")
        if (contract.required_version is not None
                and act.payload.get("version") != contract.required_version):
            failures.append("artifact version mismatch")
    elif contract.kind == "action":
        if act.procedure_receipt_id is None:
            failures.append(
                "action requires a procedure receipt; an agent saying it "
                "performed the action is not evidence")
        if act.procedure_status not in (None, "completed"):
            failures.append(
                f"procedure status {act.procedure_status!r} cannot discharge "
                f"an action obligation")
        unproven = [c for c in contract.postconditions
                    if c not in act.postconditions_proven]
        if unproven:
            failures.append(f"unproven postconditions {unproven}")
    elif contract.kind == "follow_up_decision":
        selected = act.payload.get("selected_option")
        if contract.selected_option_required and selected is None:
            failures.append("decision act selected no option")
        elif selected is not None and selected not in contract.options:
            failures.append(
                f"selected option {selected!r} is not in the menu "
                f"{list(contract.options)}")

    if not act.evidence and contract.kind != "follow_up_decision":
        failures.append("no evidence attached")

    if act.evidence:
        newest = max(e.observed_at for e in act.evidence)
        version = next((e.source_version for e in act.evidence
                        if e.source_version is not None), act.source_version)
        state = freshness_state(obligation.freshness, observed_at=newest,
                                now=now, source_version=version)
        if state == ROTTEN:
            failures.append(
                "evidence is rotten; a live answer cannot be replayed later "
                "as though it still describes current state")
    elif obligation.freshness.sensitivity != "none":
        failures.append("freshness-sensitive contract with no evidence")

    if not dependencies_discharged:
        failures.append("required dependencies are not discharged")

    if acceptance_predicate is not None and not acceptance_predicate(
            obligation, act):
        failures.append("acceptance predicate failed")

    return SatisfactionCheck(not failures, tuple(failures))


# =========================================================================
# The deterministic post-answer closure gate.
# =========================================================================

DISCHARGED_OUTCOME = "discharged"
FALSE_CLOSE = "false_close"
NEEDS_REVIEW = "needs_review"
REJECTED = "rejected"

_DISPOSITIONS = ("close", "follow_up_needed", "action_needed")


@dataclass(frozen=True)
class ImpactReport:
    changed_claims: Tuple[str, ...] = ()
    changed_state_refs: Tuple[str, ...] = ()
    dependency_effects: Tuple[str, ...] = ()
    plan_delta: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()

    @property
    def is_material(self) -> bool:
        return bool(self.changed_state_refs or self.dependency_effects
                    or self.plan_delta)


@dataclass(frozen=True)
class ProposedObligation:
    kind: str
    title: str
    contract: Any
    owner: str = "agent"
    priority_class: str = P2
    reason: str = ""


@dataclass(frozen=True)
class ClosureAttempt:
    obligation_id: str
    act: Act
    model_disposition: str
    impact: ImpactReport = field(default_factory=ImpactReport)
    proposed_children: Tuple[ProposedObligation, ...] = ()

    def __post_init__(self):
        if self.model_disposition not in _DISPOSITIONS:
            raise ObligationError(
                f"disposition must be one of {_DISPOSITIONS}, got "
                f"{self.model_disposition!r}")


@dataclass(frozen=True)
class ClosureResult:
    outcome: str
    obligation_id: str
    reasons: Tuple[str, ...] = ()
    child_ids: Tuple[str, ...] = ()
    required_children: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome == DISCHARGED_OUTCOME


# =========================================================================
# Park-and-return records (absorption capture).
# =========================================================================

PARK_KINDS = ("idea", "question", "course_change", "new_task")
INSTRUCTION_MODES = ("DIRECTIVE_NOW", "DIRECTIVE_LATER", "MUSING",
                     "AMBIGUOUS")
PARK_STATES = ("parked", "ready", "servicing", "satisfied", "escalated",
               "cancelled", "quarantined")
PARK_TERMINAL_STATES = frozenset({"satisfied", "escalated", "cancelled"})
RETURN_DISPOSITIONS = ("answer", "promote_to_lane", "merge_course_change",
                       "ask_clarification", "escalate")

#: The ONLY mode that may be granted preemption authority. A 1B model can
#: propose the task; it can never grant this.
_PREEMPTIVE_MODES = frozenset({"DIRECTIVE_NOW"})


@dataclass(frozen=True)
class ParkedItem:
    park_id: str
    session_id: str
    workspace_id: str
    source_event_id: str
    raw_message: str
    captured_at: int
    kind: str
    instruction_mode: str
    priority_class: str
    relation: str = "unknown"
    relation_confidence_num: int = 0       # exact rational, no floats
    relation_confidence_den: int = 100
    active_lane_id: str = ""
    active_checkpoint_ref: str = ""
    context_capsule: Mapping[str, Any] = field(default_factory=dict)
    capture_completeness: str = "complete"
    return_obligation_id: str = ""
    deadline_policy: str = SESSION_REVIEW
    state: str = "parked"
    terminal_receipt_ref: str = ""

    def __post_init__(self):
        if self.kind not in PARK_KINDS:
            raise ObligationError(f"unknown park kind {self.kind!r}")
        if self.instruction_mode not in INSTRUCTION_MODES:
            raise ObligationError(
                f"unknown instruction mode {self.instruction_mode!r}")
        if self.state not in PARK_STATES:
            raise ObligationError(f"unknown park state {self.state!r}")
        if self.capture_completeness not in ("complete", "partial", "missing"):
            raise ObligationError("unknown capture_completeness")
        _require_micros(self.captured_at, "captured_at")

    @property
    def preemption_allowed(self) -> bool:
        """Ambiguity NEVER carries preemption authority, and neither does
        recency, interest, or a confident classification."""
        return self.instruction_mode in _PREEMPTIVE_MODES

    @property
    def raw_message_digest(self) -> str:
        return cse1_digest({"raw": self.raw_message})


@dataclass(frozen=True)
class ParkReceipt:
    park_id: str
    obligation_id: str
    committed: bool
    acknowledgment: str
    deadline_policy: str


@dataclass(frozen=True)
class AgingReport:
    now: int
    aging_park_ids: Tuple[str, ...]
    overdue_park_ids: Tuple[str, ...]
    oldest_age_micros: int
    alarm: bool
    invariant_failed: bool
    message: str


@dataclass(frozen=True)
class SessionReceipt:
    session_id: str
    status: str                            # complete | blocked | parked
    opened_count: int
    discharged_count: int
    superseded_count: int
    cancelled_count: int
    open_obligations: Tuple[Mapping[str, Any], ...]
    oldest_open_at: Optional[int]
    rotten_count: int
    blocked_count: int
    waiting_for_human_count: int
    false_close_attempts: int
    promise_count: int
    unfulfilled_promise_count: int
    parks_total: int
    parks_serviced: int
    parks_escalated: int
    parks_overdue: int
    parking_invariant: str                 # "PASSED" | "FAILED"
    generated_at: int


class SessionCompletionBlocked(ObligationError):
    """Raised by `complete_session` when the ledger disagrees with 'done'."""

    def __init__(self, receipt: SessionReceipt):
        self.receipt = receipt
        super().__init__(
            f"session {receipt.session_id} has "
            f"{len(receipt.open_obligations)} non-terminal obligations")


# =========================================================================
# THE ONE SCHEDULER SCORE (W0-4, second half).
# =========================================================================

def service_score(obligation: Obligation, *, now: int,
                  age_cap_micros: int = 3600 * MICROS_PER_SECOND,
                  fairness_credit_cap: int = 200) -> int:
    """One formula. One integer. See the module docstring for why this SHAPE
    won and where the other document's two terms went.

        priority_class + deadline_pressure + bounded_age
        + fairness_credit - recent_service_cost

    Aging raises the score but can NEVER convert a musing into a directive:
    preemption authority is a separate boolean (`ParkedItem.preemption_allowed`)
    that this number does not feed.
    """
    _require_micros(now, "now")

    # priority_class, with dependency_criticality folded in (ledger §17 term 1)
    priority = _PRIORITY_WEIGHT[obligation.priority_class]
    priority += 25 * max(0, obligation.dependency_criticality)

    # deadline_pressure, with promise_breach_risk folded in (ledger §17 term 2)
    pressure = {NEXT_CHECKPOINT: 300, END_CURRENT_LANE: 120,
                SESSION_REVIEW: 30}[obligation.deadline_policy]
    if obligation.due_at is not None and now >= obligation.due_at:
        pressure += 400                     # overdue
    pressure += 40 * max(0, obligation.promise_breach_risk)

    # bounded_age — bounded on purpose; unbounded aging starves everything else
    age = min(obligation.age_micros(now), age_cap_micros)
    bounded_age = (age * 150) // max(1, age_cap_micros)

    # freshness risk rides the age component rather than forming a second sum
    if freshness_state(obligation.freshness,
                       observed_at=obligation.evidence_observed_at, now=now,
                       source_version=obligation.evidence_source_version
                       ) == ROTTEN and obligation.freshness.sensitivity != "none":
        bounded_age += 100

    # fairness_credit — never serviced yet earns the most
    fairness = min(fairness_credit_cap,
                   fairness_credit_cap // (1 + obligation.service_count))

    # recent_service_cost — just serviced, step aside
    cost = 0
    if obligation.last_serviced_at is not None:
        since = max(0, now - obligation.last_serviced_at)
        cost = max(0, 150 - (since * 150) // max(1, age_cap_micros))

    return priority + pressure + bounded_age + fairness - cost


# =========================================================================
# The procedure seam. `enumerate_obligations` is a PRODUCER, not a rival.
# =========================================================================

def families_for_task(task_kind: str,
                      surfaces: Sequence[str] = ()) -> Tuple[str, ...]:
    """Thin adapter over the SHIPPING function. No second list lives here.

    If you are tempted to write the families out in this module, stop: that is
    exactly the fork W0-4 exists to prevent.
    """
    return enumerate_obligations(task_kind, surfaces)


def register_procedure_families(ledger: "ObligationLedger", *,
                                task_id: str, task_kind: str,
                                surfaces: Sequence[str] = (),
                                workspace_id: str, session_id: str,
                                now: int,
                                owner: str = "agent") -> Tuple[str, ...]:
    """Feed procedure's coverage contract INTO the one ledger.

    Each family becomes an `Obligation(kind=PROCEDURE_FAMILY)` whose satisfying
    act is an artifact: the family's coverage record. A family with no
    hypothesis therefore shows up as an OPEN liability on the session receipt
    instead of quietly evaporating — which is procedure.py's own stated intent
    ("reported as residual uncertainty rather than dropped"), now enforced by
    the same gate as every other obligation.
    """
    ids: List[str] = []
    for family in families_for_task(task_kind, surfaces):
        obligation = Obligation(
            id=f"{task_id}:family:{family}",
            workspace_id=workspace_id,
            session_id=session_id,
            kind=PROCEDURE_FAMILY,
            title=f"cover obligation family {family!r}",
            contract=ArtifactContract(
                artifact_type="obligation_family_coverage",
                recipient="procedure_receipt",
                acceptance_predicate_ref=f"family:{family}"),
            origin_event_id=f"{task_id}:enumerate_obligations",
            owner=owner,
            priority_class=P2,
            deadline_policy=END_CURRENT_LANE,
            normalized_intent=f"cover:{family}",
            target_ref=task_id,
            span_hash=cse1_digest({"task": task_id, "family": family}),
            first_seen_at=now,
            created_at=now,
            last_progress_at=now,
            procedure_family=family,
        )
        ids.append(ledger.register(obligation, now=now).id)
    return tuple(ids)


# =========================================================================
# The ledger.
# =========================================================================

class ObligationLedger:
    """Append-only liability register with materialized obligation state.

    `events` is the source of truth for what happened; the obligation map is a
    projection. The park registry is likewise a PROJECTION — never a second
    source of truth (the park doc says so explicitly and it is right).
    """

    def __init__(self, *, workspace_id: str, session_id: str) -> None:
        if not workspace_id or not session_id:
            raise ObligationError("workspace_id and session_id are required")
        self.workspace_id = workspace_id
        self.session_id = session_id
        self._obligations: Dict[str, Obligation] = {}
        self._by_registration_key: Dict[str, str] = {}
        self._by_canonical_key: Dict[str, str] = {}
        self._parks: Dict[str, ParkedItem] = {}
        self._acts: Dict[str, Act] = {}
        self.events: List[Dict[str, Any]] = []
        self.false_close_attempts = 0
        self._park_seq = 0

    # ---------------------------------------------------------- plumbing
    def _emit(self, name: str, **payload: Any) -> None:
        if name not in EVENT_NAMES:
            raise ObligationError(f"unknown event name {name!r}")
        # Keep the local event's historical payload while never presenting the
        # journal-reserved ``kind`` spelling to an emit-like boundary.
        if "obligation_kind" in payload:
            payload["kind"] = payload.pop("obligation_kind")
        self.events.append({"event": name, "seq": len(self.events),
                            **payload})

    def _require(self, obligation_id: str) -> Obligation:
        if obligation_id not in self._obligations:
            raise ObligationError(f"unknown obligation {obligation_id!r}")
        return self._obligations[obligation_id]

    def get(self, obligation_id: str) -> Obligation:
        return self._require(obligation_id)

    def all(self) -> Tuple[Obligation, ...]:
        return tuple(self._obligations[k] for k in sorted(self._obligations))

    def open_obligations(self) -> Tuple[Obligation, ...]:
        return tuple(o for o in self.all() if o.is_open)

    def planned_artifact_obligations(self) -> Tuple[Obligation, ...]:
        """Return the planned-artifact subtype from this one ledger.

        The adapter in ``kernel_planned_artifacts`` owns resolution.  This
        projection exists here so completion and scheduler callers do not need
        a parallel obligation collection.
        """
        return tuple(o for o in self.all() if o.kind == PLANNED_ARTIFACT)

    def attach_planned_artifact_ledger(self, adapter: Any) -> None:
        """Attach the opt-in resolution view without creating another ledger."""
        self._planned_artifact_ledger = adapter

    def fulfill_planned_artifact(self, obligation_id: str, *, now: int,
                                 evidence_ref: str = "") -> Obligation:
        """Host-discharge a planned artifact after its predicates are proven.

        This narrow reducer seam is intentionally not an ``Act`` shortcut for
        ordinary work.  The planned-artifact adapter has already evaluated
        static predicates; behavior remains a separate predicate there.
        """
        obligation = self._require(obligation_id)
        if obligation.kind != PLANNED_ARTIFACT:
            raise ObligationError("fulfill_planned_artifact requires planned_artifact")
        adapter = getattr(self, "_planned_artifact_ledger", None)
        if adapter is not None:
            record = getattr(adapter, "artifacts", {}).get(obligation_id)
            if record is None or not getattr(record, "is_fulfilled", False):
                raise ObligationError(
                    "planned_artifact cannot discharge before host predicates are satisfied")
        if obligation.is_terminal:
            return obligation
        out = self._set(obligation_id, status=DISCHARGED,
                        satisfying_act_id=evidence_ref or "planned-artifact",
                        last_progress_at=now, evidence_observed_at=now)
        self._emit("PLANNED_ARTIFACT_FULFILLED", obligation_id=obligation_id,
                   evidence_ref=evidence_ref)
        return out

    def parks(self) -> Tuple[ParkedItem, ...]:
        return tuple(self._parks[k] for k in sorted(self._parks))

    def event_names(self) -> Tuple[str, ...]:
        return tuple(e["event"] for e in self.events)

    # ------------------------------------------------------- registration
    def register(self, obligation: Obligation, *, now: int,
                 allow_semantic_merge: bool = True) -> Obligation:
        """Two levels of idempotency. Neither ever resets age."""
        _require_micros(now, "now")
        if obligation.workspace_id != self.workspace_id:
            raise ObligationError(
                "cross-workspace registration is a tenancy violation")

        reg_key = obligation.registration_key()
        existing_id = self._by_registration_key.get(reg_key)
        if existing_id is not None:
            existing = self._obligations[existing_id]
            self._emit("OBLIGATION_DEDUPLICATED", obligation_id=existing_id,
                       level="exact")
            return existing

        if allow_semantic_merge and obligation.kind not in NEVER_MERGE_KINDS:
            canon = obligation.canonical_key()
            twin_id = self._by_canonical_key.get(canon)
            if twin_id is not None:
                twin = self._obligations[twin_id]
                # A NEW ask after the previous one was discharged is a NEW
                # liability, never a resurrection of the old one.
                if not twin.is_terminal:
                    merged = replace(
                        twin,
                        registration_count=twin.registration_count + 1,
                        occurrence_refs=twin.occurrence_refs
                        + (obligation.origin_event_id,),
                        last_progress_at=now,
                        # first_seen_at DELIBERATELY untouched: repeating an
                        # ask raises urgency, it does not make the original
                        # liability young again.
                    )
                    self._obligations[twin_id] = merged
                    self._by_registration_key[reg_key] = twin_id
                    self._emit("OBLIGATION_DEDUPLICATED",
                               obligation_id=twin_id, level="semantic")
                    return merged

        if obligation.id in self._obligations:
            raise ObligationError(
                f"obligation id {obligation.id!r} already exists with a "
                f"different registration key")
        stored = replace(obligation,
                         first_seen_at=obligation.first_seen_at or now,
                         created_at=obligation.created_at or now,
                         last_progress_at=obligation.last_progress_at or now)
        self._obligations[stored.id] = stored
        self._by_registration_key[reg_key] = stored.id
        self._by_canonical_key.setdefault(stored.canonical_key(), stored.id)
        self._emit("OBLIGATION_REGISTERED", obligation_id=stored.id,
                   obligation_kind=stored.kind)
        if stored.kind == AGENT_PROMISE:
            self._emit("PROMISE_REGISTERED", obligation_id=stored.id)
        return stored

    def register_promise(self, obligation: Obligation, *, now: int,
                         emit_frame: Callable[[], None]) -> Obligation:
        """Promise registration happens BEFORE the narration frame becomes
        visible. If registration raises, the frame is never emitted — an
        unregistered promise must not reach the user."""
        if obligation.kind != AGENT_PROMISE:
            raise ObligationError("register_promise is for agent promises")
        stored = self.register(obligation, now=now)
        emit_frame()
        return stored

    # --------------------------------------------------------- transitions
    def _set(self, obligation_id: str, **changes: Any) -> Obligation:
        current = self._require(obligation_id)
        updated = replace(current, **changes)
        self._obligations[obligation_id] = updated
        return updated

    def mark_blocked(self, obligation_id: str, *, now: int) -> Obligation:
        out = self._set(obligation_id, status=BLOCKED, last_progress_at=now)
        self._emit("OBLIGATION_BLOCKED", obligation_id=obligation_id)
        return out

    def claim(self, obligation_id: str, *, now: int) -> Obligation:
        obligation = self._require(obligation_id)
        if obligation.is_terminal:
            raise ObligationError("cannot claim a terminal obligation")
        out = self._set(obligation_id, status=IN_PROGRESS,
                        last_progress_at=now, last_serviced_at=now,
                        service_count=obligation.service_count + 1)
        self._emit("OBLIGATION_CLAIMED", obligation_id=obligation_id)
        return out

    def release_lease(self, obligation_id: str, *, now: int) -> Obligation:
        """A lease expiring or failing returns work to OPEN. A lease NEVER
        implies satisfaction."""
        out = self._set(obligation_id, status=OPEN, last_progress_at=now)
        return out

    def cancel(self, obligation_id: str, *, now: int, reason: str,
               authorized_by: str) -> Obligation:
        if not reason or not authorized_by:
            raise ObligationError(
                "cancellation requires an explicit reason AND an authorizing "
                "actor; silent abandonment is the thing this prevents")
        out = self._set(obligation_id, status=CANCELLED, cancel_reason=reason,
                        cancel_authorized_by=authorized_by,
                        last_progress_at=now)
        self._emit("OBLIGATION_CANCELLED", obligation_id=obligation_id,
                   reason=reason, authorized_by=authorized_by)
        return out

    def supersede(self, obligation_id: str, *, by_id: str, now: int,
                  authorized_by: str) -> Obligation:
        if not authorized_by:
            raise ObligationError("supersession requires authorization")
        self._require(by_id)
        out = self._set(obligation_id, status=SUPERSEDED,
                        last_progress_at=now,
                        dependencies=self._require(obligation_id).dependencies
                        + (DependencyEdge("supersedes", by_id),))
        self._emit("OBLIGATION_SUPERSEDED", obligation_id=obligation_id,
                   by_id=by_id)
        return out

    def rot_sweep(self, *, now: int) -> Tuple[str, ...]:
        """Rot check. Runs at resume, read, claim, dispatch, satisfaction and
        completion. A rotten obligation stays OPEN and gets a re-derivation
        work item; rot never discharges, cancels, or replaces anything."""
        rotten: List[str] = []
        for obligation in self.all():
            if not obligation.is_open:
                continue
            state = freshness_state(
                obligation.freshness,
                observed_at=obligation.evidence_observed_at, now=now,
                source_version=obligation.evidence_source_version)
            if state == ROTTEN and obligation.freshness.sensitivity != "none":
                rotten.append(obligation.id)
                self._emit("OBLIGATION_ROTTEN", obligation_id=obligation.id)
                self._emit("RE_DERIVATION_REQUESTED",
                           obligation_id=obligation.id)
        return tuple(rotten)

    # ------------------------------------------------- the closure gate
    def required_children(self, obligation: Obligation, act: Act,
                          impact: ImpactReport) -> Tuple[str, ...]:
        """Recompute required follow-ups from TYPED state, not from the
        model's prose. This is the half the model does not get a vote on."""
        required: List[str] = []
        for ref in impact.changed_state_refs:
            required.append(f"action:{ref}")
        for effect in impact.dependency_effects:
            required.append(f"decision:{effect}")
        for delta in impact.plan_delta:
            required.append(f"decision:plan:{delta}")
        for dep in obligation.dependencies:
            if dep.kind == "feeds":
                # A fed downstream obligation must be re-evaluated; if the act
                # tripped its condition the downstream work is now required.
                if act.payload.get("condition_tripped"):
                    required.append(f"action:{dep.target_id}")
        return tuple(sorted(set(required)))

    def attempt_discharge(self, attempt: ClosureAttempt, *, now: int,
                          dependencies_discharged: bool = True,
                          acceptance_predicate: Optional[
                              Callable[[Obligation, Act], bool]] = None
                          ) -> ClosureResult:
        """The nine-step gate. Answering and closing are separate operations.

        Parent discharge and child creation happen in ONE transaction: there is
        no window in which the parent is closed and the follow-up is missing.
        """
        _require_micros(now, "now")
        obligation = self._require(attempt.obligation_id)
        act = attempt.act
        self._acts[act.id] = act
        self._emit("ACT_RECORDED", obligation_id=obligation.id, act_id=act.id)

        # 2-3. validate the act against the contract
        check = satisfies(obligation, act, now=now,
                          dependencies_discharged=dependencies_discharged,
                          acceptance_predicate=acceptance_predicate)
        if not check.ok:
            self._emit("CLOSURE_GATE_EVALUATED", obligation_id=obligation.id,
                       outcome=REJECTED)
            return ClosureResult(REJECTED, obligation.id, check.failures)

        # 5. unknown impact fails closed
        if attempt.impact.unknowns:
            self._emit("CLOSURE_GATE_EVALUATED", obligation_id=obligation.id,
                       outcome=NEEDS_REVIEW)
            return ClosureResult(
                NEEDS_REVIEW, obligation.id,
                (f"impact has unresolved unknowns "
                 f"{list(attempt.impact.unknowns)}; unknown impact fails "
                 f"closed",))

        # 4/6. recompute required children from typed state
        required = self.required_children(obligation, act, attempt.impact)

        # 7-8. disposition arbitration
        if required and attempt.model_disposition == "close":
            self.false_close_attempts += 1
            self._emit("FALSE_CLOSE_REJECTED", obligation_id=obligation.id,
                       required=list(required))
            self._emit("CLOSURE_GATE_EVALUATED", obligation_id=obligation.id,
                       outcome=FALSE_CLOSE)
            return ClosureResult(
                FALSE_CLOSE, obligation.id,
                ("obligation answered, but the resulting state requires "
                 "tracked follow-up",) + required,
                required_children=len(required))
        if not required and attempt.model_disposition != "close":
            # A disposition claiming follow-up with nothing to follow up on is
            # noise, not safety: it would mint an untracked child.
            if not attempt.proposed_children:
                self._emit("CLOSURE_GATE_EVALUATED",
                           obligation_id=obligation.id, outcome=REJECTED)
                return ClosureResult(
                    REJECTED, obligation.id,
                    (f"disposition {attempt.model_disposition!r} proposes no "
                     f"children and the recomputed impact requires none",))
        if required and len(attempt.proposed_children) < len(required):
            self._emit("CLOSURE_GATE_EVALUATED", obligation_id=obligation.id,
                       outcome=REJECTED)
            return ClosureResult(
                REJECTED, obligation.id,
                (f"{len(required)} child obligations are required, "
                 f"{len(attempt.proposed_children)} proposed",),
                required_children=len(required))

        # ---- atomic: create children AND discharge the parent
        child_ids: List[str] = []
        for index, proposal in enumerate(attempt.proposed_children):
            child = Obligation(
                id=f"{obligation.id}:child:{index}",
                workspace_id=self.workspace_id,
                session_id=self.session_id,
                kind=proposal.kind,
                title=proposal.title,
                contract=proposal.contract,
                origin_event_id=act.id,
                owner=proposal.owner,
                priority_class=proposal.priority_class,
                deadline_policy=END_CURRENT_LANE,
                normalized_intent=proposal.title,
                span_hash=cse1_digest({"parent": obligation.id,
                                       "index": index}),
                first_seen_at=now, created_at=now, last_progress_at=now,
                parent_id=obligation.id,
                dependencies=(DependencyEdge("spawned_by", obligation.id),),
            )
            self.register(child, now=now, allow_semantic_merge=False)
            child_ids.append(child.id)
            self._emit("FOLLOW_UP_REGISTERED", obligation_id=child.id,
                       parent_id=obligation.id)

        newest = max((e.observed_at for e in act.evidence), default=None)
        version = next((e.source_version for e in act.evidence
                        if e.source_version is not None), act.source_version)
        self._set(obligation.id, status=DISCHARGED,
                  satisfying_act_id=act.id, last_progress_at=now,
                  evidence_observed_at=newest,
                  evidence_source_version=version)
        self._emit("OBLIGATION_DISCHARGED", obligation_id=obligation.id,
                   act_id=act.id)
        self._emit("CLOSURE_GATE_EVALUATED", obligation_id=obligation.id,
                   outcome=DISCHARGED_OUTCOME)

        # 9. dependency re-evaluation: a discharged source unblocks its
        # dependents; an `invalidates` edge rots them instead.
        for other in self.all():
            for dep in other.dependencies:
                if dep.target_id != obligation.id or not other.is_open:
                    continue
                if dep.kind in ("requires", "blocks") and other.status == BLOCKED:
                    self._set(other.id, status=OPEN, last_progress_at=now)
                elif dep.kind == "invalidates":
                    self._set(other.id, evidence_observed_at=None)
                    self._emit("OBLIGATION_ROTTEN", obligation_id=other.id)

        return ClosureResult(DISCHARGED_OUTCOME, obligation.id, (),
                             tuple(child_ids), len(required))

    # ------------------------------------------------------ park & return
    def park(self, item: ParkedItem, *, now: int,
             acknowledge: Callable[[ParkReceipt], None] | None = None
             ) -> ParkReceipt:
        """Absorption. ONE durable parked item and EXACTLY ONE return
        obligation, committed together. The acknowledgment is sent only after
        the transaction commits — if persistence fails we must not claim the
        item was parked."""
        _require_micros(now, "now")
        if item.park_id in self._parks:
            raise ObligationError(f"park {item.park_id!r} already exists")
        obligation_id = f"{item.park_id}:return"
        obligation = Obligation(
            id=obligation_id,
            workspace_id=self.workspace_id,
            session_id=self.session_id,
            kind=RETURN,
            title=f"service parked {item.kind}: {item.raw_message[:60]}",
            contract=FollowUpDecisionContract(
                decision_ref=item.park_id,
                decision_maker="agent",
                options=RETURN_DISPOSITIONS,
                expected_actor="agent"),
            origin_event_id=item.source_event_id,
            owner="agent",
            priority_class=item.priority_class,
            deadline_policy=item.deadline_policy,
            normalized_intent=item.raw_message_digest,
            span_hash=item.raw_message_digest,
            first_seen_at=now, created_at=now, last_progress_at=now,
            park_id=item.park_id,
        )
        self._emit("PARK_INTENT", park_id=item.park_id)
        stored_item = replace(item, return_obligation_id=obligation_id)
        self._parks[item.park_id] = stored_item
        self.register(obligation, now=now, allow_semantic_merge=False)
        self._emit("PARK_RECEIPT", park_id=item.park_id,
                   obligation_id=obligation_id)

        receipt = ParkReceipt(
            park_id=item.park_id, obligation_id=obligation_id, committed=True,
            acknowledgment=_park_acknowledgment(stored_item),
            deadline_policy=item.deadline_policy)
        if acknowledge is not None:
            acknowledge(receipt)
        return receipt

    def service_park(self, park_id: str, *, disposition: str, now: int,
                     act: Optional[Act] = None,
                     escalate_to: str = "",
                     next_deadline: Optional[int] = None) -> ParkedItem:
        """A park ends in service, escalation, or explicit cancellation.
        "Still parked" is never a successful terminal state."""
        if park_id not in self._parks:
            raise ObligationError(f"unknown park {park_id!r}")
        if disposition not in RETURN_DISPOSITIONS:
            raise ObligationError(
                f"disposition must be one of {RETURN_DISPOSITIONS}")
        item = self._parks[park_id]
        obligation = self._require(item.return_obligation_id)

        if disposition == "escalate":
            if not escalate_to or next_deadline is None:
                raise ObligationError(
                    "an escalation needs a visible owner AND a new deadline; "
                    "a silent carry-forward is not a terminal state")
            self._set(obligation.id, status=WAITING_FOR_HUMAN, owner="human",
                      due_at=next_deadline, last_progress_at=now)
            self._parks[park_id] = replace(item, state="escalated",
                                           terminal_receipt_ref=escalate_to)
            self._emit("PARK_ESCALATED", park_id=park_id, owner=escalate_to)
            return self._parks[park_id]

        if disposition == "ask_clarification":
            self._set(obligation.id, status=WAITING_FOR_HUMAN, owner="human",
                      last_progress_at=now)
            self._parks[park_id] = replace(item, state="servicing")
            return self._parks[park_id]

        satisfying = act or Act(
            id=f"{park_id}:decision", kind="decision_recorded", actor="agent",
            occurred_at=now, payload={"selected_option": disposition},
            evidence=(Evidence(ref=f"park:{park_id}", observed_at=now),))
        result = self.attempt_discharge(
            ClosureAttempt(obligation_id=obligation.id, act=satisfying,
                           model_disposition="close"), now=now)
        if not result.ok:
            raise GateError(
                f"park service rejected by the closure gate: "
                f"{'; '.join(result.reasons)}")
        self._parks[park_id] = replace(item, state="satisfied",
                                       terminal_receipt_ref=satisfying.id)
        return self._parks[park_id]

    def cancel_park(self, park_id: str, *, now: int, reason: str,
                    authorized_by: str) -> ParkedItem:
        item = self._parks[park_id]
        self.cancel(item.return_obligation_id, now=now, reason=reason,
                    authorized_by=authorized_by)
        self._parks[park_id] = replace(item, state="cancelled")
        self._emit("PARK_CANCELLED", park_id=park_id)
        return self._parks[park_id]

    def aging_report(self, *, now: int,
                     aging_after_micros: int = 600 * MICROS_PER_SECOND,
                     alarm_threshold: int = 3) -> AgingReport:
        """Fail-loud aging. Three aging items is an operational alarm; ONE
        overdue P0/P1 is already an invariant failure."""
        _require_micros(now, "now")
        aging: List[str] = []
        overdue: List[str] = []
        oldest = 0
        for item in self.parks():
            if item.state in PARK_TERMINAL_STATES:
                continue
            obligation = self._obligations.get(item.return_obligation_id)
            if obligation is None or obligation.is_terminal:
                continue
            age = now - item.captured_at
            oldest = max(oldest, age)
            if age > aging_after_micros:
                aging.append(item.park_id)
            due = obligation.due_at
            past_due = due is not None and now >= due
            if item.priority_class in (P0, P1) and (
                    past_due or age > aging_after_micros):
                overdue.append(item.park_id)
        alarm = len(aging) >= alarm_threshold
        invariant_failed = bool(overdue)
        if alarm:
            self._emit("PARK_AGING_ALARM", count=len(aging))
        message = (
            f"PARK_AGING_ALARM: {len(aging)} parked items require service; "
            f"oldest age {oldest // MICROS_PER_SECOND}s" if alarm
            else f"{len(aging)} aging parked items")
        if invariant_failed:
            message = (f"PARKING_INVARIANT: FAILED unserviced={overdue} " +
                       message)
        return AgingReport(now=now, aging_park_ids=tuple(sorted(aging)),
                           overdue_park_ids=tuple(sorted(overdue)),
                           oldest_age_micros=oldest, alarm=alarm,
                           invariant_failed=invariant_failed,
                           message=message)

    # -------------------------------------------------- session lifecycle
    def park_session(self, *, now: int) -> SessionReceipt:
        """Parking is a PERSISTENCE operation, not a completion operation.
        Every obligation stays open; the session status is `parked`."""
        self._emit("SESSION_PARKED", now=now)
        return self.session_receipt(now=now, status_override="parked")

    def resume_session(self, *, now: int) -> Tuple[str, ...]:
        """On return the LEDGER is authoritative: recompute rot, re-evaluate
        dependencies, and surface aged/blocked work before any new work."""
        self._emit("SESSION_RESUMED", now=now)
        return self.rot_sweep(now=now)

    def session_receipt(self, *, now: int,
                        status_override: Optional[str] = None
                        ) -> SessionReceipt:
        _require_micros(now, "now")
        entries: List[Mapping[str, Any]] = []
        rotten = blocked = waiting = 0
        for obligation in self.all():
            if not obligation.is_open:
                continue
            state = freshness_state(
                obligation.freshness,
                observed_at=obligation.evidence_observed_at, now=now,
                source_version=obligation.evidence_source_version)
            if state == ROTTEN and obligation.freshness.sensitivity != "none":
                rotten += 1
            if obligation.status == BLOCKED:
                blocked += 1
            if obligation.status == WAITING_FOR_HUMAN:
                waiting += 1
            entries.append({
                "obligation_id": obligation.id,
                "kind": obligation.kind,
                "title": obligation.title,
                "age_micros": obligation.age_micros(now),
                "age_band": _age_band(obligation.age_micros(now)),
                "freshness_state": state,
                "status": obligation.status,
                "owner": obligation.owner,
                "next_required_act": obligation.required_act_kind,
                "blocked_by": [d.target_id for d in obligation.dependencies
                               if d.kind in ("requires", "blocks")],
            })
        promises = [o for o in self.all() if o.kind == AGENT_PROMISE]
        parks = self.parks()
        overdue = self.aging_report(now=now).overdue_park_ids
        unserviced = [p for p in parks
                      if p.state not in PARK_TERMINAL_STATES]
        status = status_override or ("complete" if not entries else "blocked")
        return SessionReceipt(
            session_id=self.session_id,
            status=status,
            opened_count=len(self._obligations),
            discharged_count=sum(1 for o in self.all()
                                 if o.status == DISCHARGED),
            superseded_count=sum(1 for o in self.all()
                                 if o.status == SUPERSEDED),
            cancelled_count=sum(1 for o in self.all()
                                if o.status == CANCELLED),
            open_obligations=tuple(entries),
            oldest_open_at=min((o.first_seen_at for o in self.all()
                                if o.is_open), default=None),
            rotten_count=rotten, blocked_count=blocked,
            waiting_for_human_count=waiting,
            false_close_attempts=self.false_close_attempts,
            promise_count=len(promises),
            unfulfilled_promise_count=sum(1 for o in promises if o.is_open),
            parks_total=len(parks),
            parks_serviced=sum(1 for p in parks if p.state == "satisfied"),
            parks_escalated=sum(1 for p in parks if p.state == "escalated"),
            parks_overdue=len(overdue),
            parking_invariant="PASSED" if not (overdue or unserviced)
            else "FAILED",
            generated_at=now)

    def complete_session(self, *, now: int) -> SessionReceipt:
        """A kernel command with no model override. If the ledger disagrees
        with "nothing else remains", the ledger wins, loudly."""
        planned_view = getattr(self, "_planned_artifact_ledger", None)
        if planned_view is not None and hasattr(planned_view, "completion_blockers"):
            # Recompute current bytes before taking the ordinary session
            # receipt, so a deleted or regressed planned input cannot remain
            # discharged merely because the prior projection was green.
            planned_view.completion_blockers(now=now)
        self.rot_sweep(now=now)
        receipt = self.session_receipt(now=now)
        if receipt.open_obligations or receipt.parking_invariant == "FAILED":
            self._emit("SESSION_COMPLETION_BLOCKED",
                       open_count=len(receipt.open_obligations))
            raise SessionCompletionBlocked(receipt)
        self._emit("SESSION_COMPLETED", now=now)
        return receipt


def _age_band(age_micros: int) -> str:
    seconds = age_micros // MICROS_PER_SECOND
    if seconds < 300:
        return "new"
    if seconds < 1800:
        return "aging"
    if seconds < 7200:
        return "old"
    return "critical"


def _park_acknowledgment(item: ParkedItem) -> str:
    """The acknowledgment names the interpretation and promises only a
    deadline that actually exists in the return obligation."""
    when = {NEXT_CHECKPOINT: "at the next checkpoint",
            END_CURRENT_LANE: "at the end of the current lane",
            SESSION_REVIEW: "during session review"}[item.deadline_policy]
    if item.instruction_mode == "AMBIGUOUS":
        return (f"I captured this as a possible {item.kind.replace('_', ' ')} "
                f"with medium confidence. I'm not switching lanes. Should it "
                f"replace the current work, or should I park it? Either way "
                f"I'll come back to it {when}.")
    if item.preemption_allowed:
        return (f"I heard an explicit directive. I'll finish this bounded "
                f"step, checkpoint it, then switch. Tracked {when}.")
    return (f"Parked as a {item.kind.replace('_', ' ')}. I'm finishing the "
            f"current lane first and will circle back {when}.")


def planned_artifact_obligations(ledger: ObligationLedger) -> Tuple[Obligation, ...]:
    """Compatibility projection for callers that only have the old ledger API."""
    if not isinstance(ledger, ObligationLedger):
        raise TypeError("planned_artifact_obligations requires an ObligationLedger")
    return ledger.planned_artifact_obligations()
