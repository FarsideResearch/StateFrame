"""PROCEDURE — the process-discipline layer wrapped around any task.

Design authority: ``/tmp/solq/w31/a_procedure_layer.md``. Where that spec named
an API this tree does not have (a ``Snapshot`` type, ``registry.enumerate``, a
``pure_decision_evaluation`` entry point, ``detect_surface``), the BUILT shape
wins and the deviation is named in a comment rather than invented.

Design rule: explicit procedures make verification reproducible. So the procedure is CODE, and the model is confined to
two seats at the table — confirming and vetoing entries on a menu that code
enumerated, and proposing candidates code then has to schedule a check for.
The model never marks an assumption verified, never marks a check passed, never
removes an observation, never declares a review complete.

Four phases, in this order, with a code-owned gate between each:

    HYPOTHESIZE  enumerate candidate causes/approaches from a versioned
                 obligation pack keyed on task kind + detected work surface,
                 plus explicit assumptions. A family with no hypothesis is
                 recorded UNCOVERED — never silently omitted.

    VERIFY       systematic elimination. Every check is a CALLABLE deterministic
                 probe whose result is recorded as a ternary observation with an
                 input/output digest. Assumptions are vetoable ledger entries; a
                 vetoed (or unresolved hard) assumption BLOCKS ACT.

    ACT          the injected actor runs. An actor failure is an OBSERVATION,
                 not an exception that escapes the procedure.

    ADVERSARIAL_REVIEW
                 review questions are generated FROM THE WORK — which files and
                 symbols the action touched, which surfaces that implies, which
                 checks came back UNKNOWN, which assumptions the action leaned
                 on. Each question carries a CHECK, and a question whose probe
                 is missing resolves UNKNOWN and BLOCKS. Questions are generated
                 before any model opinion is sought, so the model can never set
                 its own exam.

Failure routing is deterministic and severity-ordered (spec section 6):
FIX_FORWARD (bounded, immutable attempts — a new attempt, never a rewind) ->
ESCALATE (information request packet) -> ABSTAIN (honest basis). A failed check
NEVER silently passes; the only path to COMPLETE is a review with zero blocking
findings and zero unresolved evidence.

Everything is offline, deterministic, and pure: no clock, no filesystem, no
network, no uuid. Ids derive from content digests and ordinals, so two runs over
the same inputs produce byte-identical receipts. ``model_hook=None`` is the
default and the whole module runs modelless.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

MODULE = "detreason.procedure"
VERSION = "1"
POLICY_VERSION = "procedure/1"
SCHEMA_VERSION = "detreason.procedure_receipt/1"


class ProcedureError(ValueError):
    """A contract violation, raised at construction time."""


# ----------------------------------------------------------------- vocabulary
TRUE, FALSE, UNKNOWN, CONFLICT = "TRUE", "FALSE", "UNKNOWN", "CONFLICT"
TERNARY = (TRUE, FALSE, UNKNOWN, CONFLICT)

HYPOTHESIZE = "HYPOTHESIZE"
VERIFY = "VERIFY"
ACT = "ACT"
ADVERSARIAL_REVIEW = "ADVERSARIAL_REVIEW"
COMPLETE = "COMPLETE"
PHASES = (HYPOTHESIZE, VERIFY, ACT, ADVERSARIAL_REVIEW, COMPLETE)

FIX_FORWARD = "FIX_FORWARD"
ESCALATE = "ESCALATE"
ABSTAIN = "ABSTAIN"
ROUTES = (FIX_FORWARD, ESCALATE, ABSTAIN, COMPLETE)

# hypothesis / assumption statuses
PENDING = "pending"
SUPPORTED = "supported"
ELIMINATED = "eliminated"
UNRESOLVED = "unresolved"
UNCOVERED = "uncovered"
VETOED = "vetoed"
VERIFIED = "verified"
REFUTED = "refuted"

BLOCKING, MAJOR, MINOR = "blocking", "major", "minor"
SEVERITIES = (BLOCKING, MAJOR, MINOR)

SRC_REGISTRY, SRC_MODEL, SRC_HUMAN = "registry", "model", "human"


# ------------------------------------------------------------------- digests
def _canonical(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # pragma: no cover - default=str makes this near-impossible
        return str(obj)


def digest(obj: Any) -> str:
    """Content-addressed ref, matching decision_event.content_ref's shape."""
    return "sha256:" + hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------- obligation registry
#: Versioned obligation packs, keyed by task kind. This is NOT a proof of
#: universal completeness; it is a deterministic COVERAGE CONTRACT, and a family
#: with no hypothesis is reported as residual uncertainty rather than dropped.
OBLIGATION_PACKS: Dict[str, Tuple[str, ...]] = {
    "bug_fix": ("reproduction", "location", "root_cause", "patch_strategy",
                "regression", "compatibility", "error_path", "security",
                "operations"),
    "feature": ("requirement_interpretation", "integration_point",
                "data_contract", "authorization", "migration",
                "failure_behavior", "observability", "compatibility", "tests"),
    "qa": ("oracle", "environment", "coverage", "negative_cases", "flakiness",
           "evidence_integrity", "reporting"),
    "question": ("claim_extraction", "source_sufficiency", "contradiction",
                 "freshness", "uncertainty", "answer_scope"),
    "judgment": ("option_enumeration", "criteria", "evidence_sufficiency",
                 "rejected_alternatives", "reversibility"),
}

#: The detected work surface ADDS obligations on top of the task-kind pack.
SURFACE_PACKS: Dict[str, Tuple[str, ...]] = {
    "database": ("migration", "constraint", "index", "lock", "data_compatibility"),
    "public_api": ("callers", "serialization", "versioning", "error_contract"),
    "queue": ("idempotency", "retry", "visibility_timeout", "dead_letter"),
    "auth": ("authorization_boundary", "tenant_isolation", "audit_trail"),
    "test": ("oracle_weakening", "fixture_drift", "coverage_loss"),
}

TASK_KINDS = tuple(sorted(OBLIGATION_PACKS))
SURFACES = tuple(sorted(SURFACE_PACKS))


def enumerate_obligations(task_kind: str,
                          surfaces: Sequence[str] = ()) -> Tuple[str, ...]:
    """The complete, ordered, de-duplicated obligation menu. Code owns this."""
    if task_kind not in OBLIGATION_PACKS:
        raise ProcedureError(
            f"unknown task_kind {task_kind!r}; known: {TASK_KINDS}")
    out: List[str] = list(OBLIGATION_PACKS[task_kind])
    for surface in surfaces:
        if surface not in SURFACE_PACKS:
            raise ProcedureError(
                f"unknown surface {surface!r}; known: {SURFACES}")
        for fam in SURFACE_PACKS[surface]:
            if fam not in out:
                out.append(fam)
    return tuple(out)


# ----------------------------------------------------------------- data types
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_kind: str
    request: str
    surfaces: Tuple[str, ...] = ()
    declared_scope: Tuple[str, ...] = ()   # files the action may touch

    def __post_init__(self):
        if not self.task_id:
            raise ProcedureError("task_id is required")
        if self.task_kind not in OBLIGATION_PACKS:
            raise ProcedureError(f"unknown task_kind {self.task_kind!r}")
        for s in self.surfaces:
            if s not in SURFACE_PACKS:
                raise ProcedureError(f"unknown surface {s!r}")

    @property
    def request_digest(self) -> str:
        return digest({"task_id": self.task_id, "request": self.request})


@dataclass
class Hypothesis:
    id: str
    family: str
    claim: str
    predicted_observation: str = ""
    check_ids: Tuple[str, ...] = ()
    status: str = PENDING
    source: str = SRC_REGISTRY
    evidence: Tuple[str, ...] = ()      # why it reached its status

    def to_dict(self) -> dict:
        return {"id": self.id, "family": self.family, "claim": self.claim,
                "predicted_observation": self.predicted_observation,
                "check_ids": list(self.check_ids), "status": self.status,
                "source": self.source, "evidence": list(self.evidence)}


@dataclass
class Assumption:
    """A vetoable ledger entry. `hard` assumptions gate ACT."""
    id: str
    claim: str
    falsifier: str = ""
    check_id: str = ""
    hard: bool = True
    status: str = PENDING
    vetoed_by: str = ""
    evidence: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"id": self.id, "claim": self.claim, "falsifier": self.falsifier,
                "check_id": self.check_id, "hard": self.hard,
                "status": self.status, "vetoed_by": self.vetoed_by,
                "evidence": list(self.evidence)}


@dataclass
class CheckSpec:
    """A deterministic probe. `probe` is a CALLABLE — never a prose instruction.

    ``probe(context: dict)`` returns one of:
      * a ternary string,
      * ``(value, evidence_refs)``,
      * ``{"value": ..., "evidence_refs": [...], "detail": "..."}``.
    Anything else, or a raise, becomes an UNKNOWN observation with the reason
    recorded. A probe that cannot be run is never a pass.
    """
    id: str
    claim: str
    probe: Optional[Callable[[dict], Any]] = None
    supports: str = ""        # hypothesis id this check discriminates
    assumption_id: str = ""   # assumption id this check verifies
    hard: bool = False
    pivotal: bool = False
    max_calls: int = 1
    kind: str = "deterministic"   # deterministic | bounded_lookup | sensor

    def to_dict(self) -> dict:
        return {"id": self.id, "claim": self.claim, "kind": self.kind,
                "supports": self.supports, "assumption_id": self.assumption_id,
                "hard": self.hard, "pivotal": self.pivotal,
                "max_calls": self.max_calls,
                "has_probe": self.probe is not None}


@dataclass(frozen=True)
class Observation:
    check_id: str
    value: str
    evidence_refs: Tuple[str, ...] = ()
    detail: str = ""
    attempt: int = 1
    input_digest: str = ""
    output_digest: str = ""
    phase: str = VERIFY

    def to_dict(self) -> dict:
        return {"check_id": self.check_id, "value": self.value,
                "evidence_refs": list(self.evidence_refs), "detail": self.detail,
                "attempt": self.attempt, "input_digest": self.input_digest,
                "output_digest": self.output_digest, "phase": self.phase}


@dataclass(frozen=True)
class ActionOutcome:
    """What an injected actor returns. A failure here is DATA, not an exception."""
    ok: bool
    artifact: str = ""
    touched_files: Tuple[str, ...] = ()
    touched_symbols: Tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class ActionRecord:
    attempt: int
    ok: bool
    artifact_digest: str
    touched_files: Tuple[str, ...]
    touched_symbols: Tuple[str, ...]
    detail: str = ""

    def to_dict(self) -> dict:
        return {"attempt": self.attempt, "ok": self.ok,
                "artifact_digest": self.artifact_digest,
                "touched_files": list(self.touched_files),
                "touched_symbols": list(self.touched_symbols),
                "detail": self.detail}


@dataclass
class ReviewQuestion:
    id: str
    generator_rule: str
    claim: str
    severity: str = BLOCKING
    probe_name: str = ""
    subject: str = ""              # the file/symbol/check that generated it

    def to_dict(self) -> dict:
        return {"id": self.id, "generator_rule": self.generator_rule,
                "claim": self.claim, "severity": self.severity,
                "probe_name": self.probe_name, "subject": self.subject}


@dataclass(frozen=True)
class Finding:
    question_id: str
    generator_rule: str
    severity: str
    value: str
    detail: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity == BLOCKING and self.value != TRUE

    def to_dict(self) -> dict:
        return {"question_id": self.question_id,
                "generator_rule": self.generator_rule,
                "severity": self.severity, "value": self.value,
                "detail": self.detail, "blocking": self.blocking}


@dataclass(frozen=True)
class Budgets:
    max_attempts: int = 2        # total attempts incl. the first; fix-forward is bounded
    max_checks: int = 200
    max_model_calls: int = 1

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ProcedureError("max_attempts must be >= 1")


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    passed: bool
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"gate": self.gate, "passed": self.passed,
                "reasons": list(self.reasons)}


# --------------------------------------------------------------- model hooks
#: The model's ONLY seat. It ranks/confirms/vetoes a menu code built, and may
#: propose additions. It cannot mutate state: the return value is parsed into a
#: typed command, unknown ids are rejected, and a veto becomes a ledger entry
#: that SCHEDULES a discriminating check rather than deleting anything.
@dataclass(frozen=True)
class MenuCommand:
    confirm: Tuple[str, ...] = ()
    veto: Tuple[str, ...] = ()
    propose: Tuple[Tuple[str, str], ...] = ()      # (family, claim)
    rejected: Tuple[str, ...] = ()                 # ids/entries code refused
    malformed: str = ""

    def to_dict(self) -> dict:
        return {"confirm": list(self.confirm), "veto": list(self.veto),
                "propose": [list(p) for p in self.propose],
                "rejected": list(self.rejected), "malformed": self.malformed}


def parse_menu_command(raw: Any, known_ids: Sequence[str],
                       known_families: Sequence[str]) -> MenuCommand:
    """Fail-closed parse of a model menu reply. Never raises."""
    if not isinstance(raw, Mapping):
        return MenuCommand(malformed=f"not-a-mapping:{type(raw).__name__}")
    unknown_keys = sorted(set(raw) - {"confirm", "veto", "propose"})
    if unknown_keys:
        return MenuCommand(malformed=f"unknown-keys:{','.join(unknown_keys)}")
    known = set(known_ids)
    fams = set(known_families)
    confirm, veto, rejected = [], [], []
    for key, sink in (("confirm", confirm), ("veto", veto)):
        vals = raw.get(key) or ()
        if isinstance(vals, str) or not isinstance(vals, Sequence):
            return MenuCommand(malformed=f"{key}-not-a-list")
        for v in vals:
            if v in known:
                if v not in sink:
                    sink.append(v)
            else:
                rejected.append(f"{key}:{v}")
    propose: List[Tuple[str, str]] = []
    raw_prop = raw.get("propose") or ()
    if isinstance(raw_prop, str) or not isinstance(raw_prop, Sequence):
        return MenuCommand(malformed="propose-not-a-list")
    for item in raw_prop:
        if not isinstance(item, Mapping):
            rejected.append("propose:not-a-mapping")
            continue
        fam, claim = item.get("family"), item.get("claim")
        if fam not in fams:
            rejected.append(f"propose:unknown-family:{fam}")
            continue
        if not isinstance(claim, str) or not claim.strip():
            rejected.append(f"propose:empty-claim:{fam}")
            continue
        propose.append((fam, claim.strip()))
    # A hypothesis both confirmed and vetoed is a contradiction, not a signal.
    both = [i for i in confirm if i in veto]
    for i in both:
        confirm.remove(i)
        veto.remove(i)
        rejected.append(f"contradiction:{i}")
    return MenuCommand(tuple(confirm), tuple(veto), tuple(propose),
                       tuple(rejected))


# ------------------------------------------------------- review question rules
#: Work-derived generator rules. Each maps a signal PRESENT IN THE ACTION to a
#: checkable question. The probe is looked up by ``probe_name`` in the injected
#: ``review_probes`` mapping; a missing probe yields UNKNOWN, which for a
#: blocking question is a FINDING. Silence is never a pass.
REVIEW_RULES: Tuple[Tuple[str, str, str, str], ...] = (
    # (rule, probe_name, severity, claim template)
    ("request_satisfied", "request_satisfied", BLOCKING,
     "The action satisfies the original request"),
    ("changed_behavior_has_oracle", "changed_behavior_has_oracle", BLOCKING,
     "Every changed behavior gained a test or an explicit oracle"),
    ("no_unknown_became_pass", "", BLOCKING,
     "No UNKNOWN observation was treated as a pass"),
    ("evidence_linkage", "", BLOCKING,
     "Every acceptance claim links to evidence from this attempt"),
    ("scope_respected", "", BLOCKING,
     "No file was touched outside the declared scope"),
    ("no_self_serving_edit", "no_self_serving_edit", BLOCKING,
     "The action did not weaken tests, fixtures, policies or evidence"),
)

#: Per-surface rules, added only when the touched work actually implies them.
SURFACE_REVIEW_RULES: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "database": (("migration_forward_safe", BLOCKING,
                  "Forward migration, constraints and indexes are present and data-compatible"),),
    "public_api": (("callers_still_typecheck", BLOCKING,
                    "Every in-repository caller still type-checks and its targeted tests pass"),),
    "queue": (("duplicate_delivery_safe", BLOCKING,
               "Duplicate delivery remains safe and retry is bounded"),),
    "auth": (("unauthorized_fails_closed", BLOCKING,
              "Unauthorized and cross-tenant cases fail closed"),),
    "test": (("oracle_not_weakened", BLOCKING,
              "The test oracle was not weakened and still fails against a seeded defect"),),
}


# ------------------------------------------------------------- probe execution
def execute_check(check: CheckSpec, context: dict, *, attempt: int,
                  phase: str = VERIFY) -> Observation:
    """Run one probe and record the result. NEVER raises, never returns a pass
    it did not observe. A missing probe is UNKNOWN, not TRUE."""
    in_digest = digest({"check": check.id, "claim": check.claim,
                        "context": sorted(context.keys())})
    if check.probe is None:
        return Observation(check.id, UNKNOWN, (), "no-probe-registered", attempt,
                           in_digest, digest(None), phase)
    try:
        raw = check.probe(context)
    except Exception as exc:  # a probe that raises is an UNKNOWN observation
        return Observation(check.id, UNKNOWN, (),
                           f"probe-error:{type(exc).__name__}:{str(exc)[:120]}",
                           attempt, in_digest, digest("error"), phase)
    value, refs, detail = _coerce_probe_result(raw)
    return Observation(check.id, value, refs, detail, attempt, in_digest,
                       digest(raw), phase)


def _coerce_probe_result(raw: Any) -> Tuple[str, Tuple[str, ...], str]:
    if isinstance(raw, str):
        return (raw, (), "") if raw in TERNARY else (UNKNOWN, (), f"bad-value:{raw[:40]}")
    if isinstance(raw, bool):
        # Deliberate: booleans are accepted because probes are frequently
        # predicates, but they are widened to the ternary vocabulary at once.
        return (TRUE if raw else FALSE), (), ""
    if isinstance(raw, Mapping):
        value = raw.get("value")
        refs = tuple(str(r) for r in (raw.get("evidence_refs") or ()))
        detail = str(raw.get("detail", ""))
        if isinstance(value, bool):
            value = TRUE if value else FALSE
        if value not in TERNARY:
            return UNKNOWN, refs, detail or f"bad-value:{str(value)[:40]}"
        return value, refs, detail
    if isinstance(raw, tuple) and len(raw) == 2:
        value, refs = raw
        if isinstance(value, bool):
            value = TRUE if value else FALSE
        if value not in TERNARY:
            return UNKNOWN, (), f"bad-value:{str(value)[:40]}"
        return value, tuple(str(r) for r in (refs or ())), ""
    return UNKNOWN, (), f"bad-result-type:{type(raw).__name__}"


# --------------------------------------------------------------------- run
@dataclass
class Attempt:
    """One IMMUTABLE pass through the machine. Fix-forward adds a new attempt;
    nothing is ever rewound, reset, or discarded."""
    number: int
    phases_run: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    checks: List[CheckSpec] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    menu_command: Optional[MenuCommand] = None
    gates: List[GateVerdict] = field(default_factory=list)
    action: Optional[ActionRecord] = None
    review_questions: List[ReviewQuestion] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    route: str = ""
    reasons: List[str] = field(default_factory=list)
    phase_reached: str = HYPOTHESIZE

    def observation_for(self, check_id: str) -> Optional[Observation]:
        for obs in self.observations:
            if obs.check_id == check_id:
                return obs
        return None

    def to_dict(self) -> dict:
        return {
            "attempt": self.number,
            "phases_run": list(self.phases_run),
            "phase_reached": self.phase_reached,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "checks": [c.to_dict() for c in self.checks],
            "observations": [o.to_dict() for o in self.observations],
            "uncovered_families": list(self.uncovered),
            "menu_command": self.menu_command.to_dict() if self.menu_command else None,
            "gates": [g.to_dict() for g in self.gates],
            "action": self.action.to_dict() if self.action else None,
            "review_questions": [q.to_dict() for q in self.review_questions],
            "findings": [f.to_dict() for f in self.findings],
            "route": self.route,
            "reasons": list(self.reasons),
        }


@dataclass
class ProcedureResult:
    task: TaskSpec
    attempts: List[Attempt]
    route: str
    terminal_phase: str
    open_items: List[str] = field(default_factory=list)
    escalation: Optional[dict] = None

    @property
    def complete(self) -> bool:
        return self.route == COMPLETE

    @property
    def last(self) -> Attempt:
        return self.attempts[-1]

    def receipt(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "module": MODULE,
            "version": VERSION,
            "policy_version": POLICY_VERSION,
            "run_id": run_id_for(self.task),
            "task_id": self.task.task_id,
            "task_kind": self.task.task_kind,
            "request_digest": self.task.request_digest,
            "surfaces": list(self.task.surfaces),
            "declared_scope": list(self.task.declared_scope),
            "obligation_menu": list(enumerate_obligations(self.task.task_kind,
                                                          self.task.surfaces)),
            "attempts": [a.to_dict() for a in self.attempts],
            "route": self.route,
            "terminal_phase": self.terminal_phase,
            "complete": self.complete,
            "open_items": list(self.open_items),
            "escalation": self.escalation,
        }


def run_id_for(task: TaskSpec) -> str:
    return digest({"policy": POLICY_VERSION, "task": task.task_id,
                   "kind": task.task_kind, "request": task.request,
                   "surfaces": list(task.surfaces)})


# ------------------------------------------------------------------ the gates
def gate_hypothesize(attempt: Attempt) -> GateVerdict:
    """HYPOTHESIZE -> VERIFY."""
    reasons: List[str] = []
    if not attempt.hypotheses:
        reasons.append("no-hypotheses-enumerated")
    for a in attempt.assumptions:
        if not a.falsifier:
            reasons.append(f"assumption-without-falsifier:{a.id}")
        if not a.check_id:
            reasons.append(f"assumption-without-check:{a.id}")
    known = {c.id for c in attempt.checks}
    for a in attempt.assumptions:
        if a.check_id and a.check_id not in known:
            reasons.append(f"assumption-check-missing:{a.id}->{a.check_id}")
    for h in attempt.hypotheses:
        for cid in h.check_ids:
            if cid not in known:
                reasons.append(f"hypothesis-check-missing:{h.id}->{cid}")
    for c in attempt.checks:
        if c.max_calls < 1:
            reasons.append(f"check-unbounded:{c.id}")
    # UNCOVERED families are recorded, not fatal — they are residual uncertainty
    # that must survive into the receipt (spec section 3).
    return GateVerdict("hypothesize->verify", not reasons, tuple(reasons))


def gate_verify(attempt: Attempt) -> GateVerdict:
    """VERIFY -> ACT. Hard assumptions pass, no pivotal UNKNOWN/CONFLICT,
    no hard check failed, no live veto."""
    reasons: List[str] = []
    for a in attempt.assumptions:
        if a.status == VETOED:
            reasons.append(f"assumption-vetoed:{a.id}:{a.vetoed_by}")
        elif a.hard and a.status != VERIFIED:
            reasons.append(f"hard-assumption-{a.status}:{a.id}")
    for c in attempt.checks:
        obs = attempt.observation_for(c.id)
        if obs is None:
            if c.hard or c.pivotal:
                reasons.append(f"check-not-run:{c.id}")
            continue
        if c.hard and obs.value != TRUE:
            reasons.append(f"hard-check-{obs.value}:{c.id}")
        if c.pivotal and obs.value in (UNKNOWN, CONFLICT):
            reasons.append(f"pivotal-{obs.value}:{c.id}")
    if all(h.status == ELIMINATED for h in attempt.hypotheses) and attempt.hypotheses:
        reasons.append("every-hypothesis-eliminated")
    return GateVerdict("verify->act", not reasons, tuple(reasons))


def gate_act(attempt: Attempt) -> GateVerdict:
    """ACT -> ADVERSARIAL_REVIEW."""
    reasons: List[str] = []
    rec = attempt.action
    if rec is None:
        reasons.append("no-action-recorded")
    else:
        if not rec.ok:
            reasons.append(f"action-failed:{rec.detail[:80]}")
        if not rec.artifact_digest:
            reasons.append("action-produced-no-artifact")
    return GateVerdict("act->review", not reasons, tuple(reasons))


def gate_review(attempt: Attempt) -> GateVerdict:
    """REVIEW -> COMPLETE."""
    reasons: List[str] = []
    answered = {f.question_id for f in attempt.findings}
    for q in attempt.review_questions:
        if q.id not in answered:
            reasons.append(f"review-question-unanswered:{q.id}")
    for f in attempt.findings:
        if f.blocking:
            # The generator rule travels in the reason so routing can tell a
            # security/contract failure (ESCALATE) from a repairable one.
            reasons.append(
                f"blocking-finding:{f.question_id}:{f.generator_rule}:{f.value}")
    return GateVerdict("review->complete", not reasons, tuple(reasons))


# -------------------------------------------------------------- fail routing
#: Reason prefixes that are never repairable in-process.
_ESCALATE_PREFIXES = ("pivotal-CONFLICT", "assumption-vetoed",
                      "human-authorization")
_ESCALATE_RULES = frozenset({"unauthorized_fails_closed", "no_self_serving_edit",
                             "migration_forward_safe"})
_ABSTAIN_PREFIXES = ("check-not-run", "no-probe", "review-question-unanswered",
                     "hard-assumption-pending", "hard-assumption-unresolved")


def route_failure(gate: GateVerdict, attempt: Attempt, budgets: Budgets, *,
                  repair_available: bool) -> Tuple[str, Tuple[str, ...]]:
    """Deterministic, severity-ordered routing (spec section 6).

    ESCALATE beats FIX_FORWARD beats ABSTAIN: a security/contract failure or an
    unresolvable conflict is never "repaired" by another swing at it.
    """
    if gate.passed:
        return COMPLETE, ()
    reasons = tuple(gate.reasons)
    escalating = tuple(r for r in reasons
                       if r.startswith(_ESCALATE_PREFIXES)
                       or any(f":{rule}" in r or r.endswith(rule)
                              for rule in _ESCALATE_RULES))
    if escalating:
        return ESCALATE, escalating
    # Repeating the same failure class without new evidence escalates rather
    # than burning another attempt on the same swing (spec section 6).
    if _repeats_prior_failure(attempt, reasons):
        return ESCALATE, reasons + ("repeat-failure-class-without-new-evidence",)
    abstaining = tuple(r for r in reasons if r.startswith(_ABSTAIN_PREFIXES))
    if abstaining:
        return ABSTAIN, abstaining
    if repair_available and attempt.number < budgets.max_attempts:
        return FIX_FORWARD, reasons
    return ABSTAIN, reasons + (("repair-budget-exhausted",)
                               if repair_available else
                               ("no-bounded-repair-available",))


def _repeats_prior_failure(attempt: Attempt, reasons: Sequence[str]) -> bool:
    prior = getattr(attempt, "_prior_failure_digest", "")
    return bool(prior) and prior == digest(sorted(_classes(reasons)))


def _classes(reasons: Sequence[str]) -> List[str]:
    return [r.split(":", 1)[0] for r in reasons]


def escalation_packet(task: TaskSpec, attempt: Attempt,
                      reasons: Sequence[str]) -> dict:
    """The human decision packet: exact finding, evidence, alternatives, ask."""
    return {
        "task_id": task.task_id,
        "attempt": attempt.number,
        "phase": attempt.phase_reached,
        "reasons": list(reasons),
        "evidence": [o.to_dict() for o in attempt.observations
                     if o.value in (FALSE, UNKNOWN, CONFLICT)],
        "alternatives": [h.to_dict() for h in attempt.hypotheses
                         if h.status in (PENDING, UNRESOLVED, SUPPORTED)],
        "required_human_input": sorted({_ask_for(r) for r in reasons}),
    }


def _ask_for(reason: str) -> str:
    head = reason.split(":", 1)[0]
    return {
        "assumption-vetoed": "Confirm or overrule the vetoed assumption",
        "pivotal-CONFLICT": "Resolve the conflicting evidence",
        "unauthorized_fails_closed": "Authorize or reject the security-relevant change",
        "no_self_serving_edit": "Review whether the change edits its own oracle",
        "migration_forward_safe": "Approve the schema migration",
        "repeat-failure-class-without-new-evidence": "Provide new evidence or a new approach",
    }.get(head, f"Adjudicate: {head}")


# ------------------------------------------------- review question generation
def derive_review_questions(task: TaskSpec, attempt: Attempt) -> List[ReviewQuestion]:
    """Questions generated FROM THE WORK PERFORMED — never a generic pack.

    Generated BEFORE any model review opinion is sought, so the model cannot
    define its own exam (spec section 5).
    """
    rec = attempt.action
    questions: List[ReviewQuestion] = []
    seq = [0]

    def add(rule: str, claim: str, severity: str, probe_name: str, subject: str):
        seq[0] += 1
        qid = f"Q{seq[0]:02d}"
        questions.append(ReviewQuestion(qid, rule, claim, severity, probe_name,
                                        subject))

    for rule, probe_name, severity, claim in REVIEW_RULES:
        add(rule, claim, severity, probe_name, task.task_id)

    if rec is not None:
        for path in rec.touched_files:
            for surface in _surfaces_for_path(path):
                for rule, severity, claim in SURFACE_REVIEW_RULES.get(surface, ()):
                    add(rule, f"{claim} ({path})", severity, rule, path)
        for sym in rec.touched_symbols:
            add("callers_still_typecheck",
                f"Every caller of {sym} still type-checks and its tests pass",
                BLOCKING, "callers_still_typecheck", sym)

    # Checks that came back UNKNOWN/CONFLICT are re-asked as review questions
    # rather than quietly aging into a pass.
    for obs in attempt.observations:
        if obs.value in (UNKNOWN, CONFLICT):
            add("unresolved_observation",
                f"Check {obs.check_id} resolved {obs.value}; it must not be read as a pass",
                MAJOR, "", obs.check_id)

    # Assumptions the action leaned on, and rejected alternatives that were close.
    for a in attempt.assumptions:
        if a.status == VERIFIED and not a.evidence:
            add("assumption_evidence_linkage",
                f"Assumption {a.id} was marked verified without linked evidence",
                BLOCKING, "", a.id)
    for h in attempt.hypotheses:
        if h.status in (PENDING, UNRESOLVED):
            add("rejected_alternative_elimination",
                f"Hypothesis {h.id} was neither supported nor eliminated",
                MAJOR, "", h.id)

    return questions


_SURFACE_PATH_HINTS: Tuple[Tuple[str, str], ...] = (
    ("migration", "database"),
    ("schema", "database"),
    ("models/", "database"),
    ("api/", "public_api"),
    ("routes", "public_api"),
    ("queue", "queue"),
    ("worker", "queue"),
    ("consumer", "queue"),
    ("auth", "auth"),
    ("permission", "auth"),
    ("test_", "test"),
    ("tests/", "test"),
    ("fixture", "test"),
    ("conftest", "test"),
)


def _surfaces_for_path(path: str) -> Tuple[str, ...]:
    low = path.lower()
    out: List[str] = []
    for hint, surface in _SURFACE_PATH_HINTS:
        if hint in low and surface not in out:
            out.append(surface)
    return tuple(out)


def execute_review(task: TaskSpec, attempt: Attempt,
                   review_probes: Mapping[str, Callable[[dict], Any]]) -> List[Finding]:
    """Answer every generated question. Built-in rules are answered from the
    RECORD (they need no probe); rule-specific questions consult the injected
    probe map. A question with no answer source resolves UNKNOWN — and an
    UNKNOWN on a blocking question is a blocking finding."""
    findings: List[Finding] = []
    for q in attempt.review_questions:
        value, detail = _answer_question(task, attempt, q, review_probes)
        findings.append(Finding(q.id, q.generator_rule, q.severity, value, detail))
    return findings


def _answer_question(task: TaskSpec, attempt: Attempt, q: ReviewQuestion,
                     probes: Mapping[str, Callable[[dict], Any]]) -> Tuple[str, str]:
    # Structural rules code can answer from the immutable record.
    if q.generator_rule == "no_unknown_became_pass":
        bad = [o.check_id for o in attempt.observations
               if o.value in (UNKNOWN, CONFLICT)
               and _check_by_id(attempt, o.check_id) is not None
               and _check_by_id(attempt, o.check_id).hard]
        return (FALSE, f"hard-checks-unknown:{','.join(bad)}") if bad else (TRUE, "")
    if q.generator_rule == "evidence_linkage":
        # VERIFY-phase observations only: review probes are answered against
        # the record and are not themselves acceptance claims, so including
        # them would let the review poison its own question.
        missing = [o.check_id for o in attempt.observations
                   if o.phase == VERIFY and o.value == TRUE
                   and not o.evidence_refs]
        return ((FALSE, f"unlinked-acceptance:{','.join(missing)}") if missing
                else (TRUE, ""))
    if q.generator_rule == "scope_respected":
        rec = attempt.action
        if rec is None:
            return UNKNOWN, "no-action"
        if not task.declared_scope:
            return UNKNOWN, "no-declared-scope"
        out = [f for f in rec.touched_files if f not in task.declared_scope]
        return (FALSE, f"outside-scope:{','.join(out)}") if out else (TRUE, "in-scope")
    if q.generator_rule == "unresolved_observation":
        return FALSE, f"{q.subject} unresolved"
    if q.generator_rule in ("assumption_evidence_linkage",
                            "rejected_alternative_elimination"):
        return FALSE, q.subject

    probe = probes.get(q.probe_name) if q.probe_name else None
    if probe is None:
        return UNKNOWN, f"no-probe:{q.probe_name or q.generator_rule}"
    spec = CheckSpec(id=q.id, claim=q.claim, probe=probe)
    obs = execute_check(spec, {"task": task, "attempt": attempt, "question": q},
                        attempt=attempt.number, phase=ADVERSARIAL_REVIEW)
    attempt.observations.append(obs)
    return obs.value, obs.detail


def _check_by_id(attempt: Attempt, check_id: str) -> Optional[CheckSpec]:
    for c in attempt.checks:
        if c.id == check_id:
            return c
    return None


# ------------------------------------------------------------ the state machine
@dataclass
class Providers:
    """Everything the procedure needs from the outside world, injected.

    All optional: with none of them the procedure still runs end to end and
    reports honestly that it could establish nothing.
    """
    #: hypothesis_source(task, family) -> Sequence[Mapping] with keys
    #: claim / predicted_observation / (optional) source.
    hypothesis_source: Optional[Callable[[TaskSpec, str], Sequence[Mapping]]] = None
    #: check_source(task, hypothesis) -> Sequence[CheckSpec]
    check_source: Optional[Callable[[TaskSpec, Hypothesis], Sequence[CheckSpec]]] = None
    #: assumption_source(task) -> Sequence[Assumption]
    assumption_source: Optional[Callable[[TaskSpec], Sequence[Assumption]]] = None
    #: actor(task, attempt) -> ActionOutcome
    actor: Optional[Callable[[TaskSpec, Attempt], ActionOutcome]] = None
    #: model_hook(menu_payload: dict) -> Mapping  (confirm/veto/propose)
    model_hook: Optional[Callable[[dict], Any]] = None
    #: review_probes: rule/probe name -> callable
    review_probes: Mapping[str, Callable[[dict], Any]] = field(default_factory=dict)
    #: repair(task, attempt) -> bool  — is a bounded repair available?
    repair_available: Optional[Callable[[TaskSpec, Attempt], bool]] = None


def _compile_hypotheses(task: TaskSpec, providers: Providers,
                        attempt: Attempt) -> None:
    families = enumerate_obligations(task.task_kind, task.surfaces)
    seq = 0
    for family in families:
        rows: Sequence[Mapping] = ()
        if providers.hypothesis_source is not None:
            try:
                rows = providers.hypothesis_source(task, family) or ()
            except Exception as exc:
                rows = ()
                attempt.reasons.append(
                    f"hypothesis-source-error:{family}:{type(exc).__name__}")
        if not rows:
            attempt.uncovered.append(family)
            continue
        for row in rows:
            seq += 1
            hid = f"H{seq:02d}"
            attempt.hypotheses.append(Hypothesis(
                id=hid, family=family,
                claim=str(row.get("claim", "")),
                predicted_observation=str(row.get("predicted_observation", "")),
                source=str(row.get("source", SRC_REGISTRY)),
            ))


def _apply_model_menu(task: TaskSpec, providers: Providers,
                      attempt: Attempt) -> None:
    """The model's one seat: confirm / veto / propose over a code-built menu."""
    if providers.model_hook is None:
        return
    payload = {
        "request_digest": task.request_digest,
        "menu": [{"id": h.id, "family": h.family, "claim": h.claim}
                 for h in attempt.hypotheses],
        "families": list(enumerate_obligations(task.task_kind, task.surfaces)),
        "uncovered_families": list(attempt.uncovered),
    }
    try:
        raw = providers.model_hook(payload)
    except Exception as exc:
        attempt.menu_command = MenuCommand(
            malformed=f"hook-error:{type(exc).__name__}")
        return
    cmd = parse_menu_command(raw, [h.id for h in attempt.hypotheses],
                             enumerate_obligations(task.task_kind, task.surfaces))
    attempt.menu_command = cmd
    if cmd.malformed:
        return
    seq = len(attempt.hypotheses)
    for family, claim in cmd.propose:
        seq += 1
        hid = f"H{seq:02d}"
        attempt.hypotheses.append(Hypothesis(id=hid, family=family, claim=claim,
                                             source=SRC_MODEL))
        if family in attempt.uncovered:
            attempt.uncovered.remove(family)
        # A model proposal is a CLAIM, so it enters the ledger as an assumption
        # that a discriminating check must settle before ACT.
        attempt.assumptions.append(Assumption(
            id=f"A-{hid}", claim=claim, hard=False,
            falsifier=f"a discriminating check refutes {hid}",
            check_id=""))
    for hid in cmd.veto:
        h = _hypothesis_by_id(attempt, hid)
        if h is None:
            continue
        # A veto NEVER deletes a hypothesis and never overrides evidence. It
        # becomes a ledger entry that schedules a discriminating check.
        attempt.assumptions.append(Assumption(
            id=f"V-{hid}", claim=f"MODEL_VETOED {hid}: {h.claim}",
            falsifier=f"a discriminating check supports {hid}",
            check_id="", hard=False, status=VETOED, vetoed_by=SRC_MODEL))


def _hypothesis_by_id(attempt: Attempt, hid: str) -> Optional[Hypothesis]:
    for h in attempt.hypotheses:
        if h.id == hid:
            return h
    return None


def _compile_checks(task: TaskSpec, providers: Providers, attempt: Attempt) -> None:
    if providers.assumption_source is not None:
        for a in providers.assumption_source(task) or ():
            attempt.assumptions.append(a)
    if providers.check_source is None:
        return
    for h in attempt.hypotheses:
        try:
            specs = providers.check_source(task, h) or ()
        except Exception as exc:
            attempt.reasons.append(
                f"check-source-error:{h.id}:{type(exc).__name__}")
            continue
        ids: List[str] = []
        for spec in specs:
            spec = replace(spec, supports=spec.supports or h.id)
            attempt.checks.append(spec)
            ids.append(spec.id)
            if spec.assumption_id:
                for a in attempt.assumptions:
                    if a.id == spec.assumption_id and not a.check_id:
                        a.check_id = spec.id
        h.check_ids = tuple(ids)


def _run_checks(attempt: Attempt, budgets: Budgets, context: dict) -> None:
    """Deterministic checks first, exactly as the spec orders them."""
    ordered = sorted(attempt.checks,
                     key=lambda c: (0 if c.kind == "deterministic" else 1, c.id))
    for check in ordered[:budgets.max_checks]:
        attempt.observations.append(
            execute_check(check, context, attempt=attempt.number))


def update_statuses(attempt: Attempt) -> None:
    """Elimination bookkeeping. Pure, code-owned, evidence-recorded."""
    for h in attempt.hypotheses:
        if not h.check_ids:
            h.status = UNCOVERED
            continue
        values = []
        evidence: List[str] = []
        for cid in h.check_ids:
            obs = attempt.observation_for(cid)
            if obs is None:
                continue
            values.append(obs.value)
            evidence.append(f"{cid}={obs.value}")
        if not values:
            h.status = UNRESOLVED
        elif FALSE in values:
            h.status = ELIMINATED
        elif CONFLICT in values:
            h.status = UNRESOLVED
        elif all(v == TRUE for v in values):
            h.status = SUPPORTED
        else:
            h.status = UNRESOLVED
        h.evidence = tuple(evidence)

    for a in attempt.assumptions:
        if a.status == VETOED:
            # Only a discriminating check can lift a veto — and only by
            # observing TRUE. Absence of evidence never lifts it.
            obs = attempt.observation_for(a.check_id) if a.check_id else None
            if obs is not None and obs.value == TRUE:
                a.status = VERIFIED
                a.evidence = (f"{a.check_id}={TRUE}",)
            continue
        obs = attempt.observation_for(a.check_id) if a.check_id else None
        if obs is None:
            a.status = PENDING
            continue
        if obs.value == TRUE:
            a.status = VERIFIED
        elif obs.value == FALSE:
            a.status = REFUTED
        else:
            a.status = UNRESOLVED
        a.evidence = (f"{a.check_id}={obs.value}",)


def _open_items(attempt: Attempt) -> List[str]:
    items: List[str] = []
    for fam in attempt.uncovered:
        items.append(f"uncovered-family:{fam}")
    for h in attempt.hypotheses:
        if h.status in (PENDING, UNRESOLVED, UNCOVERED):
            items.append(f"hypothesis-{h.status}:{h.id}")
    for a in attempt.assumptions:
        if a.status in (PENDING, UNRESOLVED, VETOED, REFUTED):
            items.append(f"assumption-{a.status}:{a.id}")
    for c in attempt.checks:
        if attempt.observation_for(c.id) is None:
            items.append(f"check-not-run:{c.id}")
    for o in attempt.observations:
        if o.value in (UNKNOWN, CONFLICT):
            items.append(f"observation-{o.value}:{o.check_id}")
    answered = {f.question_id for f in attempt.findings}
    for q in attempt.review_questions:
        if q.id not in answered:
            items.append(f"review-unanswered:{q.id}")
    for f in attempt.findings:
        if f.blocking:
            items.append(f"blocking-finding:{f.question_id}")
    return items


def run_procedure(task: TaskSpec, providers: Optional[Providers] = None, *,
                  budgets: Budgets = Budgets()) -> ProcedureResult:
    """Run the full machine. Never raises on provider failure — a provider that
    blows up becomes an observation and an honest partial."""
    providers = providers or Providers()
    attempts: List[Attempt] = []
    prior_digest = ""
    route = ABSTAIN
    escalation: Optional[dict] = None
    terminal = HYPOTHESIZE

    for n in range(1, budgets.max_attempts + 1):
        attempt = Attempt(number=n)
        setattr(attempt, "_prior_failure_digest", prior_digest)
        attempts.append(attempt)

        # ---------------------------------------------------- HYPOTHESIZE
        attempt.phases_run.append(HYPOTHESIZE)
        attempt.phase_reached = HYPOTHESIZE
        _compile_hypotheses(task, providers, attempt)
        _apply_model_menu(task, providers, attempt)
        _compile_checks(task, providers, attempt)
        gate = gate_hypothesize(attempt)
        attempt.gates.append(gate)
        if not gate.passed:
            route, reasons = route_failure(gate, attempt, budgets,
                                           repair_available=False)
            attempt.route, attempt.reasons = route, list(reasons)
            terminal = HYPOTHESIZE
            prior_digest = digest(sorted(_classes(gate.reasons)))
            if route == ESCALATE:
                escalation = escalation_packet(task, attempt, reasons)
            break

        # --------------------------------------------------------- VERIFY
        attempt.phases_run.append(VERIFY)
        attempt.phase_reached = VERIFY
        _run_checks(attempt, budgets, {"task": task, "attempt": attempt})
        update_statuses(attempt)
        gate = gate_verify(attempt)
        attempt.gates.append(gate)
        if not gate.passed:
            repair = _repair_available(providers, task, attempt)
            route, reasons = route_failure(gate, attempt, budgets,
                                           repair_available=repair)
            attempt.route, attempt.reasons = route, list(reasons)
            terminal = VERIFY
            prior_digest = digest(sorted(_classes(gate.reasons)))
            if route == ESCALATE:
                escalation = escalation_packet(task, attempt, reasons)
            if route == FIX_FORWARD:
                continue
            break

        # ------------------------------------------------------------ ACT
        attempt.phases_run.append(ACT)
        attempt.phase_reached = ACT
        outcome = _run_actor(providers, task, attempt)
        attempt.action = ActionRecord(
            attempt=n, ok=outcome.ok,
            artifact_digest=digest(outcome.artifact) if outcome.artifact else "",
            touched_files=tuple(outcome.touched_files),
            touched_symbols=tuple(outcome.touched_symbols),
            detail=outcome.detail)
        gate = gate_act(attempt)
        attempt.gates.append(gate)
        if not gate.passed:
            repair = _repair_available(providers, task, attempt)
            route, reasons = route_failure(gate, attempt, budgets,
                                           repair_available=repair)
            attempt.route, attempt.reasons = route, list(reasons)
            terminal = ACT
            prior_digest = digest(sorted(_classes(gate.reasons)))
            if route == ESCALATE:
                escalation = escalation_packet(task, attempt, reasons)
            if route == FIX_FORWARD:
                continue
            break

        # ------------------------------------------- ADVERSARIAL_REVIEW
        attempt.phases_run.append(ADVERSARIAL_REVIEW)
        attempt.phase_reached = ADVERSARIAL_REVIEW
        attempt.review_questions = derive_review_questions(task, attempt)
        attempt.findings = execute_review(task, attempt, providers.review_probes)
        gate = gate_review(attempt)
        attempt.gates.append(gate)
        if not gate.passed:
            repair = _repair_available(providers, task, attempt)
            route, reasons = route_failure(gate, attempt, budgets,
                                           repair_available=repair)
            attempt.route, attempt.reasons = route, list(reasons)
            terminal = ADVERSARIAL_REVIEW
            prior_digest = digest(sorted(_classes(gate.reasons)))
            if route == ESCALATE:
                escalation = escalation_packet(task, attempt, reasons)
            if route == FIX_FORWARD:
                continue
            break

        attempt.phases_run.append(COMPLETE)
        attempt.phase_reached = COMPLETE
        attempt.route = route = COMPLETE
        terminal = COMPLETE
        break

    result = ProcedureResult(task=task, attempts=attempts, route=route,
                             terminal_phase=terminal,
                             open_items=_open_items(attempts[-1]),
                             escalation=escalation)
    return result


def _run_actor(providers: Providers, task: TaskSpec,
               attempt: Attempt) -> ActionOutcome:
    if providers.actor is None:
        return ActionOutcome(False, detail="no-actor-registered")
    try:
        out = providers.actor(task, attempt)
    except Exception as exc:
        # An actor failure is an OBSERVATION, not an exception that bypasses
        # the procedure (spec section 1, ACT->REVIEW guard).
        return ActionOutcome(False,
                             detail=f"actor-error:{type(exc).__name__}:{str(exc)[:120]}")
    if not isinstance(out, ActionOutcome):
        return ActionOutcome(False, detail=f"bad-actor-result:{type(out).__name__}")
    return out


def _repair_available(providers: Providers, task: TaskSpec,
                      attempt: Attempt) -> bool:
    if providers.repair_available is None:
        return False
    try:
        return bool(providers.repair_available(task, attempt))
    except Exception:
        return False


# ------------------------------------------------------------ receipt audit
def verify_receipt(receipt: Mapping) -> List[str]:
    """Audit a receipt WITHOUT re-running anything. Returns problem strings;
    empty means the record itself certifies what it claims."""
    problems: List[str] = []
    if receipt.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema-version-mismatch")
    for key in ("run_id", "task_id", "task_kind", "request_digest", "attempts",
                "route", "terminal_phase", "obligation_menu"):
        if key not in receipt:
            problems.append(f"missing-key:{key}")
    if problems:
        return problems
    if receipt["route"] not in ROUTES:
        problems.append(f"unknown-route:{receipt['route']}")
    attempts = receipt.get("attempts") or []
    if not attempts:
        problems.append("no-attempts-recorded")
        return problems
    numbers = [a.get("attempt") for a in attempts]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"attempt-numbers-not-contiguous:{numbers}")

    menu = set(receipt["obligation_menu"])
    for a in attempts:
        covered = {h["family"] for h in a.get("hypotheses", ())}
        uncovered = set(a.get("uncovered_families", ()))
        missing = menu - covered - uncovered
        if missing:
            problems.append(
                f"attempt{a['attempt']}:families-silently-omitted:{sorted(missing)}")
        obs_ids = {o["check_id"] for o in a.get("observations", ())}
        for h in a.get("hypotheses", ()):
            if h["status"] == ELIMINATED and not h.get("evidence"):
                problems.append(f"eliminated-without-evidence:{h['id']}")
            if h["status"] == SUPPORTED and not h.get("evidence"):
                problems.append(f"supported-without-evidence:{h['id']}")
        for asm in a.get("assumptions", ()):
            if asm["status"] == VERIFIED and not asm.get("evidence"):
                problems.append(f"assumption-verified-without-evidence:{asm['id']}")
            if asm["status"] == VERIFIED and asm.get("check_id") not in obs_ids:
                problems.append(f"assumption-verified-without-observation:{asm['id']}")
        answered = {f["question_id"] for f in a.get("findings", ())}
        for q in a.get("review_questions", ()):
            if q["id"] not in answered:
                problems.append(f"review-question-unanswered:{q['id']}")

    last = attempts[-1]
    if receipt["route"] == COMPLETE:
        if last.get("phase_reached") != COMPLETE:
            problems.append("complete-without-reaching-complete-phase")
        if not last.get("review_questions"):
            problems.append("complete-without-review-questions")
        for f in last.get("findings", ()):
            if f.get("blocking"):
                problems.append(f"complete-with-blocking-finding:{f['question_id']}")
        if receipt.get("open_items"):
            problems.append(f"complete-with-open-items:{receipt['open_items'][:3]}")
    if receipt["route"] == ESCALATE and not receipt.get("escalation"):
        problems.append("escalate-without-packet")
    if receipt["route"] in (ABSTAIN, ESCALATE) and not last.get("reasons"):
        problems.append("terminal-route-without-reasons")
    return problems


def partial_report(result: ProcedureResult) -> dict:
    """Honest partial: exactly which phase we stopped in and what is still open."""
    last = result.last
    return {
        "task_id": result.task.task_id,
        "complete": result.complete,
        "route": result.route,
        "phase_reached": last.phase_reached,
        "phases_run": list(last.phases_run),
        "attempts_used": len(result.attempts),
        "open_items": list(result.open_items),
        "reasons": list(last.reasons),
        "uncovered_families": list(last.uncovered),
        "escalation": result.escalation,
    }
