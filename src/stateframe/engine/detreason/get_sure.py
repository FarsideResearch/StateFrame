"""GET-SURE — the bounded abstention-to-sure engine.

Design authority: sol_answers w63 `a_getsure_engine.md` (executor + action
catalog + re-decide-from-scratch), `a_sure_semantics.md` (doubt -> retirement
evidence table, risk ladder, independence accounting, evidence floors) and
`a_getsure_measure.md` (the false-sure rate is the sacred metric).

The U-ledger (`detreason/uledger.py`) stays authoritative for typed doubt rows,
the accounting identity, `ready_to_act` and `next_action`. This module adds the
three things the ledger deliberately does not do:

1. EXECUTION. `next_action` names the cheapest-biggest-dU move; nothing ran it.
   The action catalog here maps each doubt type to executable acquisition
   callables built on substrate that already exists in this tree
   (`detreason/knowledge.py`, `witness_replay.py`, `test_gen_oracle.py`,
   corpus/git records, the seam graph). Actions that need the FROZEN fixer are
   declared as honest stubs carrying their exact wiring seam — they are never
   selected, and a run that needs one says so in its abstention.

2. EVIDENCE THAT CANNOT LIE ABOUT ITSELF. Every acquisition emits an
   `EvidenceRecord` addressed as a `SnapshotV1` body (`detreason/snapshot.py`),
   and NO executor may retire a row. Executors report observations; the
   code-owned closure predicates in this file decide retirement, then hand the
   U-ledger a `ClosureRecord` that must independently satisfy the ledger's own
   criterion validation. Three separate gates, none of which trusts the model.

3. RE-DECIDE FROM SCRATCH. After evidence lands, the injected `decide` callable
   is handed base facts + accepted evidence + the consumed-action ledger +
   remaining budget, and NOTHING ELSE. It never sees the previous verdict or
   the previous ledger, so evidence is free to expose NEW doubts instead of
   silently confirming an earlier optimistic read. `ReDecideRequest` enforces
   that structurally (there is no field to smuggle a prior verdict through).

THE THREE LOAD-BEARING REFUSALS
-------------------------------
* An executor returning ``{"retired": true}`` retires nothing. Only
  `check_closure()` retires, and only when the type-specific predicate passes.
* Correlated evidence counts ONCE. Two runs of the same test in the same
  environment are one channel; a `thin_margin` row therefore cannot be closed
  by repetition, no matter how many times it succeeds.
* Model self-report is inadmissible at ADMISSION, not merely down-weighted.
  Evidence whose channel method family is ``model`` is recorded `invalid`.

WHAT ``SURE`` MEANS HERE (stated loudly, because it is the whole bet)
--------------------------------------------------------------------
The w63 engine doc's invariant is ``open_doubt_count == 0``. This tree's
stopping predicate is `ULedger.ready_to_act(policy)`, which blocks on hard-max
violations first and on weighted exposure second. Those agree exactly at the
`IRREVERSIBLE` policy (hard_max 0.0 for unknown_atom / ungrounded_symbol), which
is what risk tiers R3/R4 select. At lower tiers `ready_to_act` can pass with
open low-exposure rows still on the ledger; a `SureResult` therefore ALWAYS
carries `open_doubt_count` and `unretired_row_ids` and never reports zero when
rows remain. Reading SURE without reading those two fields is a misuse.

THE SEAM — exact wiring for the gym/fixer pass (nothing below is live yet)
--------------------------------------------------------------------------
`fixer.py`, `probes/*` and `probes/unified_gym.py` are FROZEN (a live board
runs on them), so this module touches none of them. Four edges are needed, and
each is a one-way read from something that already exists:

1. LEDGER -> ENGINE. `probes/unified_gym.py:cycle_uledger` already builds the
   per-cycle `ULedger` and calls `ready_to_act()` / `next_action()`. The engine
   takes that same `ULedger` object:
       gs.run_get_sure(run_id=cycle_id, ledger=led,
                       context=gs.GetSureContext(...), tier=gs.R2)
   No new projection is required — `uledger_rows_from_stream` already produces
   rows whose ids carry the subject that `_target_for` parses.

2. ENGINE -> RE-DECIDE. `decide` is injected. The fixer's re-run is the live
   implementation: build a fresh span/pamphlet from
   `ReDecideRequest.context.base_facts` PLUS the accepted evidence, call the
   fixer's normal entry point, project its certificate back through
   `event_projections.uledger_rows_from_stream`, and return the new `ULedger`.
   The request deliberately carries no prior verdict, so the fixer re-derives
   rather than being told what to conclude.

3. WITNESS + GENERATED-TEST INPUTS. Both executors are IMPLEMENTED but need the
   span the fixer holds. Populate
   `GetSureContext.witness_inputs = {"fn_source", "span_lines", "candidates",
   "symptom_class", "fixture_id"}` from the cert the gym already enriches. Until
   that is fed they return `no_witness_inputs` — an unfed action, deliberately
   distinguishable from an unimplemented one.

4. THE TWO STUBS. `repro_flip` and `instrumentation_probe` need the fixer's
   apply/rollback path and the `test_gen_oracle._tracer` exposed as a standalone
   probe respectively; the full wiring text lives on their catalog entries
   (`CATALOG[REPRO_FLIP].seam`) and is reprinted verbatim into any priced
   abstention that wanted them.

Measured against the recorded ledger (see `replay_cycle_uledger`), the ledger
recommends only three distinct action names, two of which map onto implemented
executors. Neither stub is on the critical path today.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

from . import snapshot as snap
from .uledger import (
    COST_CLASSES, DETERMINISTIC_CHECK, EVIDENCE_SPAN,
    FALSIFICATION_PROBE, INDEPENDENT_AUDIT, IRREVERSIBLE, PREDICTED_RISK,
    REVERSIBLE, STANDARD, THIN_MARGIN, UNEXPLORED_BRANCH, UNGROUNDED_SYMBOL,
    UNKNOWN_ATOM, UNVERIFIED_CLAIM, ClosureRecord, ClosureRejected, ULedger,
    URow,
)

__all__ = [
    # vocabulary
    "USAGE_INDEX_LOOKUP", "KNOWLEDGE_CHAIN", "WITNESS_EXECUTION",
    "GENERATED_DISCRIMINATING_TEST", "REPRO_FLIP", "INSTRUMENTATION_PROBE",
    "CORPUS_GIT_EVIDENCE", "SEAM_GRAPH_WALK", "ASK_HUMAN", "ACTION_KINDS",
    "RETIRE", "CONTRIBUTE", "FORBIDDEN", "RETIREMENT_MATRIX",
    "R0", "R1", "R2", "R3", "R4", "RISK_TIERS", "RiskLadderEntry",
    "SUPPORT", "INCONCLUSIVE", "INVALID", "CONTRADICTION",
    # independence
    "Channel", "effective_channels", "correlation_classes",
    # evidence
    "EvidenceRecord", "ExecutorResult", "CoverageReceipt", "MarginReceipt",
    # catalog
    "ActionCatalogEntry", "CATALOG", "catalog_digest", "action_for_name",
    "ACTION_NAME_MAP", "UNMAPPED",
    # closure
    "ClosureVerdict", "check_closure", "to_closure_record",
    # loop
    "GetSureBudget", "BudgetState", "SpendLedger", "doubt_lineage_id",
    "canonical_subject",
    "GetSureContext", "ReDecideRequest", "DecisionRecord", "ActionRequest",
    "SureResult", "PricedAbstention", "ContradictionEscalation",
    "run_get_sure", "SURE", "PRICED_ABSTENTION", "ESCALATE_CONTRADICTION",
    # measurement
    "MeasurementRecorder", "one_sided_upper_bound_95", "measurement_report",
    # replay
    "replay_cycle_uledger", "format_replay_table",
    "GetSureError",
]


class GetSureError(RuntimeError):
    """Structural misuse of the get-sure engine."""


# =========================================================================
# 1. Vocabulary
# =========================================================================

USAGE_INDEX_LOOKUP = "usage_index_lookup"
KNOWLEDGE_CHAIN = "knowledge_chain"
WITNESS_EXECUTION = "witness_execution"
GENERATED_DISCRIMINATING_TEST = "generated_discriminating_test"
REPRO_FLIP = "repro_flip"
INSTRUMENTATION_PROBE = "instrumentation_probe"
CORPUS_GIT_EVIDENCE = "corpus_git_evidence"
SEAM_GRAPH_WALK = "seam_graph_walk"
ASK_HUMAN = "ask_human"

ACTION_KINDS = (USAGE_INDEX_LOOKUP, KNOWLEDGE_CHAIN, WITNESS_EXECUTION,
                GENERATED_DISCRIMINATING_TEST, REPRO_FLIP,
                INSTRUMENTATION_PROBE, CORPUS_GIT_EVIDENCE, SEAM_GRAPH_WALK,
                ASK_HUMAN)

# Retirement power of an action over a doubt type (engine doc section 4 table).
RETIRE = "retire"          # may retire when its closure predicate passes
CONTRIBUTE = "contribute"  # evidence only; never sufficient alone
FORBIDDEN = "forbidden"    # cannot bear on this type at all

# Evidence assessments. There is deliberately no "retired" member: an executor
# reports what it saw, the closure predicate decides what that buys.
SUPPORT = "support"
INCONCLUSIVE = "inconclusive"
INVALID = "invalid"
CONTRADICTION = "contradiction"
ASSESSMENTS = (SUPPORT, INCONCLUSIVE, INVALID, CONTRADICTION)

# Evidence strength floors (semantics doc).
WEAK, WITNESSED, STRONG, EXACT, AUTHORITATIVE = (
    "weak", "witnessed", "strong", "exact", "authoritative")
_STRENGTH_ORDER = {WEAK: 0, WITNESSED: 1, STRONG: 2, EXACT: 3,
                   AUTHORITATIVE: 3}

# Method families. `model` is inadmissible by construction.
MF_STATIC_INDEX = "static_index"
MF_KNOWLEDGE = "knowledge"
MF_RUNTIME_WITNESS = "runtime_witness"
MF_GENERATED_TEST = "generated_test"
MF_CORPUS = "corpus"
MF_SEAM = "seam"
MF_HUMAN = "human"
MF_MODEL = "model"

# Cost classes -> the ledger's own cost units, so a get-sure spend and a
# ReductionAction estimate are denominated identically.
C0, C1, C2, C3, C4 = "C0", "C1", "C2", "C3", "C4"
COST_CLASS_UNITS = {
    C0: COST_CLASSES["free"],
    C1: COST_CLASSES["cheap"],
    C2: COST_CLASSES["moderate"],
    C3: COST_CLASSES["expensive"],
    C4: COST_CLASSES["human"],
}
# Milli-units keep every budget number an int, because snapshot bodies ban
# floats and a budget that cannot be content-addressed cannot be certified.
COST_CLASS_MILLI = {k: int(round(v * 1000)) for k, v in COST_CLASS_UNITS.items()}

SURE = "SURE"
PRICED_ABSTENTION = "PRICED_ABSTENTION"
ESCALATE_CONTRADICTION = "ESCALATE_CONTRADICTION"

# The engine doc's retirement matrix, extended with this tree's sixth utype.
# `ungrounded_symbol` is a naming/binding unknown, so it retires exactly where
# `unknown_atom` does; the extension is recorded here rather than assumed.
_M = {
    USAGE_INDEX_LOOKUP: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: CONTRIBUTE, UNEXPLORED_BRANCH: RETIRE,
        THIN_MARGIN: FORBIDDEN, PREDICTED_RISK: CONTRIBUTE},
    KNOWLEDGE_CHAIN: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: CONTRIBUTE,
        THIN_MARGIN: FORBIDDEN, PREDICTED_RISK: CONTRIBUTE},
    WITNESS_EXECUTION: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: RETIRE,
        THIN_MARGIN: RETIRE, PREDICTED_RISK: RETIRE},
    GENERATED_DISCRIMINATING_TEST: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: RETIRE,
        THIN_MARGIN: RETIRE, PREDICTED_RISK: RETIRE},
    REPRO_FLIP: {
        UNKNOWN_ATOM: CONTRIBUTE, UNGROUNDED_SYMBOL: CONTRIBUTE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: CONTRIBUTE,
        THIN_MARGIN: CONTRIBUTE, PREDICTED_RISK: RETIRE},
    INSTRUMENTATION_PROBE: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: CONTRIBUTE, UNEXPLORED_BRANCH: RETIRE,
        THIN_MARGIN: RETIRE, PREDICTED_RISK: RETIRE},
    CORPUS_GIT_EVIDENCE: {
        UNKNOWN_ATOM: CONTRIBUTE, UNGROUNDED_SYMBOL: CONTRIBUTE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: CONTRIBUTE,
        THIN_MARGIN: FORBIDDEN, PREDICTED_RISK: CONTRIBUTE},
    SEAM_GRAPH_WALK: {
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: CONTRIBUTE,
        THIN_MARGIN: FORBIDDEN, PREDICTED_RISK: RETIRE},
    ASK_HUMAN: {
        # Starred in the doc: a human may supply an AUTHORITATIVE answer for an
        # atom or a claim. Human approval is never a path to SURE for a margin
        # or a predicted risk — that is a waiver, and waivers escalate.
        UNKNOWN_ATOM: RETIRE, UNGROUNDED_SYMBOL: RETIRE,
        UNVERIFIED_CLAIM: RETIRE, UNEXPLORED_BRANCH: CONTRIBUTE,
        THIN_MARGIN: FORBIDDEN, PREDICTED_RISK: FORBIDDEN},
}
RETIREMENT_MATRIX = {k: dict(v) for k, v in _M.items()}


def retirement_power(action_kind: str, utype: str) -> str:
    row = RETIREMENT_MATRIX.get(action_kind)
    if row is None:
        raise GetSureError(f"unknown action_kind {action_kind!r}")
    if utype not in row:
        raise GetSureError(f"unknown utype {utype!r}")
    return row[utype]


# =========================================================================
# 2. Risk ladder (semantics doc sections "risk-proportional SURE ladder" and
#    "evidence-strength floors"). Configuration, not code branches.
# =========================================================================

R0, R1, R2, R3, R4 = "R0", "R1", "R2", "R3", "R4"


@dataclass(frozen=True)
class RiskLadderEntry:
    """What a tier demands before a doubt of a given class may retire."""
    tier: str
    label: str
    policy_name: str                 # which StakesPolicy the ledger runs under
    min_strength: str                # evidence-strength floor
    min_channels: int                # independent channels for ordinary rows
    min_channels_critical: int       # ... for severity >= critical_severity
    critical_severity: float = 0.8
    require_runtime_for_claims: bool = False
    allow_human_attestation: bool = True


RISK_TIERS = {
    R0: RiskLadderEntry(R0, "communicate", "reversible", WEAK, 1, 1,
                        allow_human_attestation=True),
    R1: RiskLadderEntry(R1, "safely experiment", "reversible", WITNESSED, 1, 1),
    R2: RiskLadderEntry(R2, "persist locally", "standard", WITNESSED, 1, 2,
                        require_runtime_for_claims=True),
    R3: RiskLadderEntry(R3, "publish externally", "irreversible", STRONG, 2, 2,
                        require_runtime_for_claims=True),
    R4: RiskLadderEntry(R4, "affect users", "irreversible", STRONG, 2, 3,
                        require_runtime_for_claims=True,
                        allow_human_attestation=False),
}

_POLICY_BY_NAME = {"reversible": REVERSIBLE, "standard": STANDARD,
                   "irreversible": IRREVERSIBLE}


def policy_for_tier(tier: str):
    """The stakes policy a tier runs the ledger under. R3/R4 -> IRREVERSIBLE,
    whose hard_max of 0.0 for unknown atoms is what makes `ready_to_act` and
    ``open_doubt_count == 0`` coincide at publish/deploy tiers."""
    entry = RISK_TIERS.get(tier)
    if entry is None:
        raise GetSureError(f"unknown risk tier {tier!r}")
    return _POLICY_BY_NAME[entry.policy_name]


# =========================================================================
# 3. Independence accounting
# =========================================================================

@dataclass(frozen=True)
class Channel:
    """Where an observation actually came from.

    The correlation class is deliberately COARSE. The semantics doc's rule is
    'if independence is uncertain, count the evidence as one', so a channel that
    declines to name its tool, environment, fixture and root observations
    collapses into its whole method family — one channel forever, however many
    times it is re-run.
    """
    channel_id: str
    method_family: str
    tool_identity: str = ""
    environment_id: str = ""
    fixture_id: str = ""
    root_observation_ids: tuple = ()

    def __post_init__(self):
        if not self.channel_id:
            raise GetSureError("channel_id is required")
        if not self.method_family:
            raise GetSureError("method_family is required")
        object.__setattr__(self, "root_observation_ids",
                           tuple(sorted(self.root_observation_ids)))

    @property
    def admissible(self) -> bool:
        """A model talking about its own output is not a channel."""
        return self.method_family != MF_MODEL

    def correlation_class(self) -> str:
        if self.root_observation_ids:
            key = {"family": self.method_family,
                   "roots": list(self.root_observation_ids)}
        elif self.tool_identity or self.environment_id or self.fixture_id:
            key = {"family": self.method_family, "tool": self.tool_identity,
                   "env": self.environment_id, "fixture": self.fixture_id}
        else:
            key = {"family": self.method_family}
        return snap.cse1_digest(key)

    def as_dict(self) -> dict:
        return {"channel_id": self.channel_id,
                "method_family": self.method_family,
                "tool_identity": self.tool_identity,
                "environment_id": self.environment_id,
                "fixture_id": self.fixture_id,
                "root_observation_ids": list(self.root_observation_ids)}


def correlation_classes(evidences) -> set:
    return {e.channel.correlation_class() for e in evidences
            if e.channel.admissible and e.assessment == SUPPORT}


def effective_channels(evidences) -> int:
    """Independent channels among SUPPORTING, admissible evidence."""
    return len(correlation_classes(evidences))


# =========================================================================
# 4. Evidence records — content-addressed, append-only, self-describing
# =========================================================================

@dataclass(frozen=True)
class CoverageReceipt:
    """Which required partitions were actually witnessed. `unexplored_branch`
    closes on this and nothing else: one green path is not coverage."""
    required: tuple = ()
    witnessed: tuple = ()
    truncation: str = "none"          # none | bounded | unknown

    def __post_init__(self):
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "witnessed", tuple(self.witnessed))
        if self.truncation not in ("none", "bounded", "unknown"):
            raise GetSureError(f"bad truncation {self.truncation!r}")

    @property
    def missing(self) -> tuple:
        return tuple(p for p in self.required if p not in set(self.witnessed))

    @property
    def complete(self) -> bool:
        return bool(self.required) and not self.missing \
            and self.truncation == "none"

    def as_dict(self) -> dict:
        return {"required": list(self.required),
                "witnessed": list(self.witnessed),
                "truncation": self.truncation,
                "missing": list(self.missing)}


@dataclass(frozen=True)
class MarginReceipt:
    """A thin-margin measurement in integer milli-units (floats are banned from
    snapshot bodies, and a margin you cannot content-address is a rumour)."""
    observed_milli: int
    limit_milli: int
    required_margin_milli: int

    def __post_init__(self):
        for f in ("observed_milli", "limit_milli", "required_margin_milli"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, int):
                raise GetSureError(f"{f} must be an int, got {v!r}")

    @property
    def margin_milli(self) -> int:
        return abs(self.observed_milli - self.limit_milli)

    @property
    def satisfied(self) -> bool:
        return self.margin_milli >= self.required_margin_milli

    def as_dict(self) -> dict:
        return {"observed_milli": self.observed_milli,
                "limit_milli": self.limit_milli,
                "required_margin_milli": self.required_margin_milli,
                "margin_milli": self.margin_milli}


@dataclass(frozen=True)
class EvidenceRecord:
    """One acquisition's result. Content-addressed via SnapshotV1.

    `assessment` is the VERIFIER's, never the executor's: `verify_result` may
    only downgrade what an executor claimed.
    """
    evidence_id: str
    action_id: str
    run_id: str
    source_decision_id: str
    doubt_lineage_ids: tuple
    action_kind: str
    channel: Channel
    strength: str
    assessment: str
    observations: tuple = ()
    citations: tuple = ()
    coverage: CoverageReceipt | None = None
    margin: MarginReceipt | None = None
    truncation: str = "none"
    detail: dict = field(default_factory=dict)
    context_digest_before: str = ""
    created_at_micros: int = 0

    def __post_init__(self):
        if self.assessment not in ASSESSMENTS:
            raise GetSureError(f"unknown assessment {self.assessment!r}")
        if self.strength not in _STRENGTH_ORDER:
            raise GetSureError(f"unknown strength {self.strength!r}")
        if self.action_kind not in ACTION_KINDS:
            raise GetSureError(f"unknown action_kind {self.action_kind!r}")
        object.__setattr__(self, "doubt_lineage_ids",
                           tuple(self.doubt_lineage_ids))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "citations", tuple(self.citations))

    def meets(self, floor: str) -> bool:
        return _STRENGTH_ORDER[self.strength] >= _STRENGTH_ORDER[floor]

    def payload(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "action_id": self.action_id,
            "run_id": self.run_id,
            "source_decision_id": self.source_decision_id,
            "doubt_lineage_ids": list(self.doubt_lineage_ids),
            "action_kind": self.action_kind,
            "channel": self.channel.as_dict(),
            "correlation_class": self.channel.correlation_class(),
            "strength": self.strength,
            "assessment": self.assessment,
            "observations": [dict(o) for o in self.observations],
            "citations": list(self.citations),
            "coverage": self.coverage.as_dict() if self.coverage else None,
            "margin": self.margin.as_dict() if self.margin else None,
            "truncation": self.truncation,
            "detail": _snapshot_safe(self.detail),
            "context_digest_before": self.context_digest_before,
            "created_at_micros": int(self.created_at_micros),
        }

    def snapshot_body(self) -> dict:
        return snap.build_snapshot(
            type_="get-sure/evidence", schema=1,
            method_id="get_sure.evidence", method_version="1",
            implementation=None,
            inputs=[snap.input_inline("action_kind", self.action_kind),
                    snap.input_inline("doubt_lineage_ids",
                                      list(self.doubt_lineage_ids))],
            execution=snap.hermetic_execution(),
            payload=self.payload())

    def digest(self) -> str:
        return snap.snapshot_hash(self.snapshot_body())


def _snapshot_safe(value):
    """Coerce a detail blob into CSE-1-representable shapes.

    Floats become exact rationals rather than being dropped: silently losing a
    measured number is precisely the silent-drop class this tree keeps getting
    bitten by.
    """
    if value is None or isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), -float("inf")):
            return {"float_unrepresentable": repr(value)}
        num, den = float(value).as_integer_ratio()
        return snap.rational(num, den)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) \
            else list(value)
        return [_snapshot_safe(v) for v in items]
    if isinstance(value, dict):
        return {str(k): _snapshot_safe(v) for k, v in value.items()}
    return repr(value)


# =========================================================================
# 5. Action catalog
# =========================================================================

@dataclass(frozen=True)
class ExecutorResult:
    """What an adapter reports. Note what is ABSENT: any way to say 'retired'."""
    observations: tuple = ()
    citations: tuple = ()
    channel: Channel | None = None
    claimed_strength: str = WEAK
    claimed_assessment: str = INCONCLUSIVE
    coverage: CoverageReceipt | None = None
    margin: MarginReceipt | None = None
    truncation: str = "none"
    detail: dict = field(default_factory=dict)
    cost_milli: int = 0
    failure_code: str = ""

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "citations", tuple(self.citations))


@dataclass(frozen=True)
class ActionCatalogEntry:
    action_kind: str
    cost_class: str
    side_effect_policy: str            # read_only | sandbox_only | human_external
    executor: object                   # callable(request, ctx) -> ExecutorResult
    implemented: bool
    method_family: str
    produces_criteria: tuple            # ULedger closure criteria it can satisfy
    seam: str = ""                      # for stubs: the exact wiring required
    catalog_version: str = "1"

    def eligible_for(self, utype: str) -> bool:
        return retirement_power(self.action_kind, utype) != FORBIDDEN

    def can_retire(self, utype: str) -> bool:
        return retirement_power(self.action_kind, utype) == RETIRE

    @property
    def cost_milli(self) -> int:
        return COST_CLASS_MILLI[self.cost_class]

    def as_dict(self) -> dict:
        return {"action_kind": self.action_kind, "cost_class": self.cost_class,
                "side_effect_policy": self.side_effect_policy,
                "implemented": self.implemented,
                "method_family": self.method_family,
                "produces_criteria": list(self.produces_criteria),
                "seam": self.seam, "catalog_version": self.catalog_version,
                "retirement": dict(RETIREMENT_MATRIX[self.action_kind])}


# ------------------------------------------------------------------ executors

def _chan(ctx, family: str, cid: str, **kw) -> Channel:
    return Channel(channel_id=cid, method_family=family,
                   environment_id=getattr(ctx, "environment_id", ""), **kw)


def exec_usage_index_lookup(request: "ActionRequest", ctx) -> ExecutorResult:
    """Query the pinned usage index (`detreason/knowledge.py`).

    Retires a statically resolvable atom. A truncated or empty index NEVER
    closes anything — 'no usage found' in an incomplete index is silence, not
    evidence (engine doc 4.1).
    """
    index = getattr(ctx, "usage_index", None)
    if index is None:
        return ExecutorResult(failure_code="no_usage_index",
                              detail={"seam": "GetSureContext.usage_index"})
    symbol = request.target.get("symbol") or ""
    if not symbol:
        return ExecutorResult(failure_code="no_symbol")
    limit = int(request.target.get("limit", 5))
    rows = index.query(symbol, limit=limit)
    total = index.count(symbol)
    truncation = "none" if total <= limit else "bounded"
    if not rows:
        return ExecutorResult(
            claimed_assessment=INCONCLUSIVE, truncation="unknown",
            channel=_chan(ctx, MF_STATIC_INDEX, f"usage:{symbol}",
                          tool_identity="knowledge.UsageIndex"),
            detail={"symbol": symbol, "matched": 0,
                    "indexed_symbols": len(index.symbols())},
            failure_code="no_matches", cost_milli=COST_CLASS_MILLI[C0])
    obs = [{"kind": "usage", "statement": r.statement,
            "path": r.path, "line": int(r.line)} for r in rows]
    cites = [r.citation() if isinstance(r.citation(), str) else str(r.citation())
             for r in rows]
    return ExecutorResult(
        observations=tuple(obs), citations=tuple(cites),
        channel=_chan(ctx, MF_STATIC_INDEX, f"usage:{symbol}",
                      tool_identity="knowledge.UsageIndex",
                      root_observation_ids=tuple(f"{r.path}:{r.line}"
                                                 for r in rows)),
        claimed_strength=EXACT if truncation == "none" else WITNESSED,
        claimed_assessment=SUPPORT, truncation=truncation,
        detail={"symbol": symbol, "matched": len(rows), "total": total},
        cost_milli=COST_CLASS_MILLI[C0])


def exec_knowledge_chain(request: "ActionRequest", ctx) -> ExecutorResult:
    """Provenance chain from authoritative facts to the claim
    (`knowledge.resolve_chain`). An exhausted budget or open terms is
    `inconclusive`, never a plausible short chain."""
    index = getattr(ctx, "usage_index", None)
    if index is None:
        return ExecutorResult(failure_code="no_usage_index",
                              detail={"seam": "GetSureContext.usage_index"})
    query = request.target.get("symbol") or request.target.get("query") or ""
    if not query:
        return ExecutorResult(failure_code="no_query")
    try:
        from .knowledge import resolve_chain
    except Exception as exc:                      # pragma: no cover - import
        return ExecutorResult(failure_code=f"import_error:{type(exc).__name__}")
    composed = resolve_chain(query, index, getattr(ctx, "knowledge_ledger", None))
    complete = bool(getattr(composed, "complete", False))
    lines = list(getattr(composed, "evidence_lines", lambda: [])())
    obs = [{"kind": "chain_line", "text": str(t)} for t in lines]
    return ExecutorResult(
        observations=tuple(obs), citations=tuple(str(t) for t in lines),
        channel=_chan(ctx, MF_KNOWLEDGE, f"knowledge:{query}",
                      tool_identity="knowledge.resolve_chain",
                      root_observation_ids=tuple(sorted(
                          str(getattr(h, "term", h))
                          for h in getattr(composed, "dependencies", ())))),
        claimed_strength=AUTHORITATIVE if complete else WEAK,
        claimed_assessment=SUPPORT if complete else INCONCLUSIVE,
        truncation="none" if complete else "bounded",
        detail={"query": query, "complete": complete,
                "lookups_used": int(getattr(composed, "lookups_used", 0)),
                "open_terms": list(getattr(composed, "open_terms", ())),
                "budget_exhausted": bool(
                    getattr(composed, "budget_exhausted", False))},
        failure_code="" if complete else "chain_incomplete",
        cost_milli=COST_CLASS_MILLI[C1])


def exec_witness_execution(request: "ActionRequest", ctx) -> ExecutorResult:
    """Run the existing deterministic witness seam (`witness_replay.py`).

    Needs the span + mutation candidates the fixer holds. When the context has
    not been fed them the result is `inconclusive` with `no_witness_inputs` —
    an unfed action, which is a different failure from an unimplemented one and
    is reported as such.
    """
    inputs = getattr(ctx, "witness_inputs", None)
    if not inputs:
        return ExecutorResult(failure_code="no_witness_inputs",
                              detail={"seam": "GetSureContext.witness_inputs"})
    try:
        import witness_replay
    except Exception as exc:
        return ExecutorResult(failure_code=f"import_error:{type(exc).__name__}",
                              detail={"seam": "witness_replay.filter_candidates"})
    span = list(inputs.get("span_lines") or [])
    muts = list(inputs.get("candidates") or [])
    task_text = inputs.get("task_text") or ""
    if len(muts) < 2:
        return ExecutorResult(failure_code="nothing_to_discriminate",
                              detail={"candidates": len(muts)})
    try:
        survivors, evidence = _call_filter_candidates(
            witness_replay, task_text, span, muts, inputs.get("cfg"))
    except Exception as exc:
        return ExecutorResult(failure_code=f"witness_error:{type(exc).__name__}",
                              cost_milli=COST_CLASS_MILLI[C2])
    n_surv = len(survivors)
    required = tuple(f"candidate:{i}" for i in range(len(muts)))
    witnessed = tuple(f"candidate:{i}" for i in range(len(muts))
                      if i not in set(_indices(survivors, muts)))
    coverage = CoverageReceipt(required=required, witnessed=witnessed,
                               truncation="none")
    discriminated = n_surv <= 1
    return ExecutorResult(
        observations=({"kind": "witness", "survivors": n_surv,
                       "candidates": len(muts)},),
        channel=_chan(ctx, MF_RUNTIME_WITNESS, f"witness:{request.action_id}",
                      tool_identity="witness_replay.filter_candidates",
                      fixture_id=str(inputs.get("fixture_id", "")),
                      root_observation_ids=tuple(sorted(
                          str(e) for e in (evidence or ())))[:16]),
        claimed_strength=WITNESSED,
        claimed_assessment=SUPPORT if discriminated else INCONCLUSIVE,
        coverage=coverage if discriminated else None,
        detail={"survivors": n_surv, "candidates": len(muts),
                "evidence": [str(e) for e in (evidence or ())][:16]},
        failure_code="" if discriminated else "witness_did_not_discriminate",
        cost_milli=COST_CLASS_MILLI[C2])


def _call_filter_candidates(mod, task_text, span, muts, cfg):
    """`witness_replay.filter_candidates` signature has drifted before; call it
    positionally with cfg only when it is accepted."""
    try:
        return mod.filter_candidates(task_text, span, muts, cfg)
    except TypeError:
        return mod.filter_candidates(task_text, span, muts)


def _indices(survivors, muts) -> list:
    out = []
    for s in survivors or ():
        for i, m in enumerate(muts):
            if m is s or m == s:
                out.append(i)
                break
    return out


def exec_generated_discriminating_test(request: "ActionRequest",
                                       ctx) -> ExecutorResult:
    """Smallest restricted test that separates the surviving hypotheses
    (`test_gen_oracle.synthesize_discriminating_test`).

    A vacuous, inert or non-discriminating test consumes the action and yields
    `inconclusive` — never a retirement (engine doc 4.4).
    """
    inputs = getattr(ctx, "witness_inputs", None)
    if not inputs:
        return ExecutorResult(failure_code="no_witness_inputs",
                              detail={"seam": "GetSureContext.witness_inputs"})
    try:
        import test_gen_oracle
    except Exception as exc:
        return ExecutorResult(
            failure_code=f"import_error:{type(exc).__name__}",
            detail={"seam": "test_gen_oracle.synthesize_discriminating_test"})
    fn_source = inputs.get("fn_source") or "\n".join(
        inputs.get("span_lines") or [])
    cands = list(inputs.get("candidates") or [])
    res = test_gen_oracle.synthesize_discriminating_test(
        fn_source, cands, inputs.get("symptom_class"),
        inputs.get("span_lines"))
    counts = dict(getattr(res, "counts", {}) or {})
    disc = list(getattr(res, "discriminating", ()) or ())
    selected = getattr(res, "selected", None)
    ok = bool(disc) and selected is not None
    required = tuple(f"case:{i}" for i in range(len(getattr(res, "cases", ()))))
    witnessed = tuple(f"case:{i}" for i in disc)
    return ExecutorResult(
        observations=({"kind": "generated_test",
                       "verdict": str(getattr(res, "verdict", "")),
                       "reason": str(getattr(res, "reason", ""))},),
        channel=_chan(ctx, MF_GENERATED_TEST, f"gentest:{request.action_id}",
                      tool_identity="test_gen_oracle",
                      fixture_id=str(inputs.get("fixture_id", "")),
                      root_observation_ids=tuple(
                          str(c) for c in getattr(res, "cases_cited", ()))[:16]),
        claimed_strength=WITNESSED,
        claimed_assessment=SUPPORT if ok else INCONCLUSIVE,
        coverage=CoverageReceipt(required=required, witnessed=witnessed,
                                 truncation="none") if ok else None,
        detail={"counts": counts, "verdict": str(getattr(res, "verdict", "")),
                "reason": str(getattr(res, "reason", "")),
                "cases_cited": [str(c)
                                for c in getattr(res, "cases_cited", ())][:16]},
        failure_code="" if ok else "no_discriminating_case",
        cost_milli=COST_CLASS_MILLI[C3])


def exec_corpus_git_evidence(request: "ActionRequest", ctx) -> ExecutorResult:
    """Pinned historical records. Historical success is provenance and prior,
    never proof about the current snapshot (engine doc 4.7), which is why the
    matrix gives it CONTRIBUTE almost everywhere."""
    corpus = getattr(ctx, "corpus", None)
    if not corpus:
        return ExecutorResult(failure_code="no_corpus",
                              detail={"seam": "GetSureContext.corpus"})
    subject = request.target.get("subject") or ""
    hits = [dict(r) for r in corpus
            if subject and subject in str(r.get("subject", ""))]
    if not hits:
        return ExecutorResult(claimed_assessment=INCONCLUSIVE,
                              failure_code="no_corpus_hits",
                              detail={"subject": subject},
                              cost_milli=COST_CLASS_MILLI[C0])
    return ExecutorResult(
        observations=tuple({"kind": "corpus", **h} for h in hits[:8]),
        citations=tuple(str(h.get("ref", "")) for h in hits[:8]),
        channel=_chan(ctx, MF_CORPUS, f"corpus:{subject}",
                      tool_identity="corpus",
                      root_observation_ids=tuple(str(h.get("ref", ""))
                                                 for h in hits[:8])),
        claimed_strength=STRONG, claimed_assessment=SUPPORT,
        detail={"subject": subject, "hits": len(hits)},
        cost_milli=COST_CLASS_MILLI[C1])


def exec_seam_graph_walk(request: "ActionRequest", ctx) -> ExecutorResult:
    """Producer/consumer walk. Retires a seam claim only when BOTH sides are
    resolved, the join key is exact and no edge is truncated (engine doc 4.8);
    an orphan stays a doubt."""
    graph = getattr(ctx, "seam_graph", None)
    if not graph:
        return ExecutorResult(failure_code="no_seam_graph",
                              detail={"seam": "GetSureContext.seam_graph"})
    key = request.target.get("seam") or request.target.get("subject") or ""
    edge = graph.get(key)
    if not edge:
        return ExecutorResult(claimed_assessment=INCONCLUSIVE,
                              failure_code="seam_not_found",
                              detail={"seam": key},
                              cost_milli=COST_CLASS_MILLI[C1])
    producer = edge.get("producer")
    consumer = edge.get("consumer")
    join_exact = bool(edge.get("join_key_exact"))
    truncated = bool(edge.get("truncated"))
    resolved = bool(producer) and bool(consumer) and join_exact and not truncated
    conflict = bool(edge.get("incompatible"))
    return ExecutorResult(
        observations=({"kind": "seam", "producer": str(producer),
                       "consumer": str(consumer),
                       "join_key_exact": join_exact},),
        citations=(str(producer or ""), str(consumer or "")),
        channel=_chan(ctx, MF_SEAM, f"seam:{key}", tool_identity="seam_graph",
                      root_observation_ids=(str(producer or ""),
                                            str(consumer or ""))),
        claimed_strength=STRONG if resolved else WEAK,
        claimed_assessment=(CONTRADICTION if conflict else
                            SUPPORT if resolved else INCONCLUSIVE),
        truncation="bounded" if truncated else "none",
        detail={"seam": key, "resolved": resolved, "incompatible": conflict},
        failure_code="" if resolved else "seam_unresolved",
        cost_milli=COST_CLASS_MILLI[C1])


def exec_ask_human(request: "ActionRequest", ctx) -> ExecutorResult:
    """Typed question with a frozen context and an allowed answer shape.

    'Looks good' retires nothing: an answer must carry an authority identity,
    a scope and supporting artifact refs, or it is `inconclusive`.
    """
    answers = getattr(ctx, "human_answers", None) or {}
    qid = request.target.get("question_id") or request.action_id
    ans = answers.get(qid)
    if ans is None:
        return ExecutorResult(claimed_assessment=INCONCLUSIVE,
                              failure_code="awaiting_human",
                              detail={"question_id": qid,
                                      "exact_question":
                                          request.target.get("question", "")},
                              cost_milli=0)
    authority = ans.get("authority") or ""
    scope = ans.get("scope") or ""
    artifacts = list(ans.get("artifact_refs") or ())
    answer = ans.get("answer")
    if not (authority and scope and artifacts and answer is not None):
        return ExecutorResult(
            claimed_assessment=INCONCLUSIVE, failure_code="unauthoritative_answer",
            detail={"question_id": qid, "has_authority": bool(authority),
                    "has_scope": bool(scope), "artifact_refs": len(artifacts)},
            cost_milli=COST_CLASS_MILLI[C4])
    return ExecutorResult(
        observations=({"kind": "human_attestation", "answer": str(answer)},),
        citations=tuple(str(a) for a in artifacts),
        channel=_chan(ctx, MF_HUMAN, f"human:{authority}",
                      tool_identity=str(authority),
                      root_observation_ids=tuple(str(a) for a in artifacts)),
        claimed_strength=AUTHORITATIVE, claimed_assessment=SUPPORT,
        detail={"question_id": qid, "authority": authority, "scope": scope},
        cost_milli=COST_CLASS_MILLI[C4])


def _stub(kind: str, seam: str):
    """An action whose substrate lives behind the FROZEN fixer.

    It returns a declared stub result rather than raising, and is marked
    `implemented=False` so eligibility never selects it. A run that would have
    needed it names it, and this seam text, in its priced abstention.
    """
    def _run(request: "ActionRequest", ctx) -> ExecutorResult:
        return ExecutorResult(claimed_assessment=INCONCLUSIVE,
                              failure_code="not_implemented",
                              detail={"action_kind": kind, "seam": seam},
                              cost_milli=0)
    _run.__name__ = f"exec_{kind}_stub"
    return _run


_REPRO_FLIP_SEAM = (
    "fixer.py holds the span, the mutation menu and the apply/rollback path. "
    "Wiring: call the fixer's per-candidate apply with base and perturbed "
    "spans, capture both certificates, and feed them here as "
    "GetSureContext.repro_pairs = [{'base_cert':..., 'perturbed_cert':..., "
    "'declared_causal_input':...}]. This module must not edit fixer.py, so the "
    "adapter belongs in the gym/fixer pass, and exec_repro_flip becomes a pure "
    "reader of those pairs.")

_INSTRUMENTATION_SEAM = (
    "test_gen_oracle._tracer already collects line/branch coverage under a "
    "hard limit, but only inside its own execution matrix. Wiring: expose that "
    "tracer as a standalone probe entry point (probe_profile, limits, deadline) "
    "and feed its per-branch edge set in as GetSureContext.probe_traces. Until "
    "the probe can certify it is behaviour-preserving under its declared "
    "profile it stays diagnostic-only and cannot retire a row.")


CATALOG = {
    USAGE_INDEX_LOOKUP: ActionCatalogEntry(
        USAGE_INDEX_LOOKUP, C0, "read_only", exec_usage_index_lookup, True,
        MF_STATIC_INDEX, (DETERMINISTIC_CHECK, EVIDENCE_SPAN)),
    KNOWLEDGE_CHAIN: ActionCatalogEntry(
        KNOWLEDGE_CHAIN, C1, "read_only", exec_knowledge_chain, True,
        MF_KNOWLEDGE, (EVIDENCE_SPAN,)),
    WITNESS_EXECUTION: ActionCatalogEntry(
        WITNESS_EXECUTION, C2, "sandbox_only", exec_witness_execution, True,
        MF_RUNTIME_WITNESS, (FALSIFICATION_PROBE, DETERMINISTIC_CHECK)),
    GENERATED_DISCRIMINATING_TEST: ActionCatalogEntry(
        GENERATED_DISCRIMINATING_TEST, C3, "sandbox_only",
        exec_generated_discriminating_test, True, MF_GENERATED_TEST,
        (FALSIFICATION_PROBE, DETERMINISTIC_CHECK)),
    CORPUS_GIT_EVIDENCE: ActionCatalogEntry(
        CORPUS_GIT_EVIDENCE, C1, "read_only", exec_corpus_git_evidence, True,
        MF_CORPUS, (EVIDENCE_SPAN,)),
    SEAM_GRAPH_WALK: ActionCatalogEntry(
        SEAM_GRAPH_WALK, C1, "read_only", exec_seam_graph_walk, True,
        MF_SEAM, (DETERMINISTIC_CHECK, EVIDENCE_SPAN)),
    ASK_HUMAN: ActionCatalogEntry(
        ASK_HUMAN, C4, "human_external", exec_ask_human, True, MF_HUMAN,
        (INDEPENDENT_AUDIT,)),
    REPRO_FLIP: ActionCatalogEntry(
        REPRO_FLIP, C2, "sandbox_only", _stub(REPRO_FLIP, _REPRO_FLIP_SEAM),
        False, MF_RUNTIME_WITNESS, (FALSIFICATION_PROBE,),
        seam=_REPRO_FLIP_SEAM),
    INSTRUMENTATION_PROBE: ActionCatalogEntry(
        INSTRUMENTATION_PROBE, C2, "sandbox_only",
        _stub(INSTRUMENTATION_PROBE, _INSTRUMENTATION_SEAM), False,
        MF_RUNTIME_WITNESS, (FALSIFICATION_PROBE,),
        seam=_INSTRUMENTATION_SEAM),
}


def catalog_digest(catalog=None) -> str:
    """Pinned identity of the catalog. A catalog change cannot alter an
    in-flight run, so the run records this digest."""
    cat = catalog or CATALOG
    return snap.cse1_digest({k: v.as_dict() for k, v in sorted(cat.items())})


# The ledger's ReductionAction names are free text written by the cert
# adapters. This is the ONE place they are mapped onto executable action kinds.
UNMAPPED = "unmapped"

ACTION_NAME_MAP = {
    # usage-index / static reads
    "grep_definitions": USAGE_INDEX_LOOKUP,
    "read_more_files": USAGE_INDEX_LOOKUP,
    "anchor_by_exact_symbol": USAGE_INDEX_LOOKUP,
    "tool_sensor_if_expressible": USAGE_INDEX_LOOKUP,
    # runtime witnesses
    "run_evidence_linkage_gate": WITNESS_EXECUTION,
    "run_witness_replay_on_diff": WITNESS_EXECUTION,
    "run_existing_test_suite": WITNESS_EXECUTION,
    # generated discrimination
    "synthesize_discriminating_witness": GENERATED_DISCRIMINATING_TEST,
    "deliberate_over_survivors": GENERATED_DISCRIMINATING_TEST,
    # probes behind the frozen fixer
    "runtime_probe": INSTRUMENTATION_PROBE,
    "paired_entailment_probe": INSTRUMENTATION_PROBE,
    "variant_verify": REPRO_FLIP,
    # humans
    "escalate_to_human": ASK_HUMAN,
    "human_review_diff": ASK_HUMAN,
    # Deliberately UNMAPPED. Re-asking a model — in a different order, or of a
    # bigger model — is the same channel with a different sentence. The
    # semantics doc counts it as one channel and it retires nothing, so it gets
    # no catalog entry rather than a flattering one.
    "reask_menu_reversed_order": UNMAPPED,
    "escalate_to_larger_model": UNMAPPED,
}


def action_for_name(name: str) -> str:
    """Ledger ReductionAction name -> catalog action kind, or UNMAPPED."""
    return ACTION_NAME_MAP.get(name, UNMAPPED)


# =========================================================================
# 6. Closure predicates — code-owned, per doubt type
# =========================================================================

@dataclass(frozen=True)
class ClosureVerdict:
    retire: bool
    criterion: str | None
    reasons: tuple
    effective_channels: int
    contradiction: bool = False
    supporting_evidence_ids: tuple = ()

    def as_dict(self) -> dict:
        return {"retire": self.retire, "criterion": self.criterion,
                "reasons": list(self.reasons),
                "effective_channels": self.effective_channels,
                "contradiction": self.contradiction,
                "supporting_evidence_ids": list(self.supporting_evidence_ids)}


def _admissible(row: URow, evidences) -> tuple:
    """Split evidence into (usable supports, reasons it was rejected)."""
    supports, reasons = [], []
    for e in evidences:
        if e.assessment == CONTRADICTION:
            continue                      # handled by the caller, not here
        if e.assessment != SUPPORT:
            reasons.append(f"{e.action_kind}: assessment={e.assessment}")
            continue
        if not e.channel.admissible:
            reasons.append(f"{e.action_kind}: model self-report is inadmissible")
            continue
        power = retirement_power(e.action_kind, row.utype)
        if power == FORBIDDEN:
            reasons.append(
                f"{e.action_kind} cannot bear on {row.utype}")
            continue
        supports.append(e)
    return supports, reasons


def _criterion_for(row: URow, supports, catalog) -> str | None:
    """Pick a ULedger closure criterion the supporting actions can actually
    produce, honouring a criterion the row itself declared."""
    produced = set()
    for e in supports:
        entry = catalog.get(e.action_kind)
        if entry is not None:
            produced |= set(entry.produces_criteria)
    if row.closure_criterion:
        return row.closure_criterion if row.closure_criterion in produced \
            else None
    for c in (DETERMINISTIC_CHECK, FALSIFICATION_PROBE, EVIDENCE_SPAN,
              INDEPENDENT_AUDIT):
        if c in produced:
            return c
    return None


def check_closure(row: URow, evidences, tier: str = R1, *, catalog=None,
                  required_partitions=()) -> ClosureVerdict:
    """THE gate. Type-specific, code-owned, and unbribable by an executor.

    Every branch below states what it refuses, because the refusals are the
    product: a retirement this function grants for the wrong reason is a
    false-sure, which is the one number the whole programme is measured on.
    """
    catalog = catalog or CATALOG
    entry = RISK_TIERS.get(tier)
    if entry is None:
        raise GetSureError(f"unknown risk tier {tier!r}")
    evidences = list(evidences)

    contradictions = [e for e in evidences if e.assessment == CONTRADICTION]
    if contradictions:
        return ClosureVerdict(
            False, None,
            tuple(f"contradiction from {e.action_kind}" for e in contradictions),
            0, contradiction=True)

    supports, reasons = _admissible(row, evidences)
    if not supports:
        return ClosureVerdict(False, None,
                              tuple(reasons or ("no supporting evidence",)), 0)

    # A retirement needs at least one action whose matrix cell is RETIRE.
    retirers = [e for e in supports if retirement_power(e.action_kind,
                                                        row.utype) == RETIRE]
    if not retirers:
        return ClosureVerdict(
            False, None,
            tuple(list(reasons) + ["only contributing evidence; no action in "
                                   "this set may retire this doubt type"]),
            effective_channels(supports))

    # Evidence-strength floor.
    strong_enough = [e for e in retirers if e.meets(entry.min_strength)]
    if not strong_enough:
        return ClosureVerdict(
            False, None,
            tuple(list(reasons) + [f"no evidence meets the {tier} strength "
                                   f"floor {entry.min_strength}"]),
            effective_channels(supports))

    # Truncation is fatal for every type: a bounded index or a partial chain
    # cannot prove an absence (engine doc 4.1).
    truncated = [e for e in strong_enough if e.truncation != "none"]
    if truncated and len(truncated) == len(strong_enough):
        return ClosureVerdict(
            False, None,
            tuple(list(reasons) + [f"{e.action_kind}: truncation="
                                   f"{e.truncation}" for e in truncated]),
            effective_channels(supports))
    strong_enough = [e for e in strong_enough if e.truncation == "none"]

    need = (entry.min_channels_critical
            if row.severity >= entry.critical_severity else entry.min_channels)

    type_reasons = []
    if row.utype in (UNKNOWN_ATOM, UNGROUNDED_SYMBOL):
        # A resolved atom needs a citation. 'no usage found' is not a value.
        cited = [e for e in strong_enough if e.citations]
        if not cited:
            type_reasons.append("unknown_atom needs a cited resolution "
                                "(authoritative source + exact location)")
        strong_enough = cited or strong_enough

    elif row.utype == UNVERIFIED_CLAIM:
        if entry.require_runtime_for_claims and row.severity >= \
                entry.critical_severity:
            fams = {e.channel.method_family for e in strong_enough}
            runtime = fams & {MF_RUNTIME_WITNESS, MF_GENERATED_TEST}
            structural = fams & {MF_STATIC_INDEX, MF_SEAM, MF_KNOWLEDGE,
                                 MF_CORPUS}
            if not (runtime and structural):
                type_reasons.append(
                    f"{tier} critical claim needs BOTH a runtime witness and "
                    f"structural/dataflow evidence; have {sorted(fams)}")

    elif row.utype == UNEXPLORED_BRANCH:
        covered = [e for e in strong_enough if e.coverage is not None]
        if not covered:
            type_reasons.append("unexplored_branch needs a coverage receipt; "
                                "a passing run is not coverage")
        else:
            required = tuple(required_partitions) or ()
            merged_witnessed = set()
            merged_required = set(required)
            complete_any = False
            for e in covered:
                merged_witnessed |= set(e.coverage.witnessed)
                merged_required |= set(e.coverage.required)
                complete_any = complete_any or e.coverage.complete
            missing = sorted(merged_required - merged_witnessed)
            if missing or not complete_any:
                detail = ", ".join(missing) if missing \
                    else "coverage receipt incomplete"
                type_reasons.append(f"partitions not witnessed: {detail}")
            strong_enough = covered

    elif row.utype == THIN_MARGIN:
        margined = [e for e in strong_enough if e.margin is not None]
        if not margined:
            type_reasons.append("thin_margin needs a measured margin receipt")
        else:
            bad = [e for e in margined if not e.margin.satisfied]
            if bad:
                type_reasons.append(
                    f"margin {bad[0].margin.margin_milli} < required "
                    f"{bad[0].margin.required_margin_milli} (milli-units)")
            strong_enough = margined
        # A margin is retired by an INDEPENDENT second channel agreeing, never
        # by repetition. This is the semantics doc's hardest rule and the one a
        # naive loop breaks first.
        need = max(need, 2)

    elif row.utype == PREDICTED_RISK:
        fams = {e.channel.method_family for e in strong_enough}
        if fams <= {MF_HUMAN}:
            type_reasons.append(
                "predicted_risk cannot be retired by human attestation alone; "
                "that is a waiver and it escalates")
        if not (fams & {MF_RUNTIME_WITNESS, MF_GENERATED_TEST, MF_SEAM}):
            type_reasons.append(
                "predicted_risk needs a risk-specific witness or a resolved "
                "seam; calibration never closes the row")

    channels = effective_channels(strong_enough)
    if channels < need:
        type_reasons.append(
            f"{channels} independent channel(s) < {need} required at {tier} "
            f"(correlated evidence counts once)")

    if type_reasons:
        return ClosureVerdict(False, None, tuple(list(reasons) + type_reasons),
                              channels)

    criterion = _criterion_for(row, strong_enough, catalog)
    if criterion is None:
        return ClosureVerdict(
            False, None,
            tuple(list(reasons) + [
                f"no supporting action can produce the declared closure "
                f"criterion {row.closure_criterion!r}"]),
            channels)
    return ClosureVerdict(True, criterion, tuple(reasons), channels,
                          supporting_evidence_ids=tuple(
                              e.evidence_id for e in strong_enough))


def to_closure_record(row: URow, verdict: ClosureVerdict, evidences, *,
                      actor: str = "get_sure") -> ClosureRecord:
    """Build the ULedger `ClosureRecord` for a granted verdict.

    The ledger validates this AGAIN with its own rules and can still reject it.
    Two independent gates over the same closure is the point, not redundancy.
    """
    if not verdict.retire or not verdict.criterion:
        raise GetSureError("to_closure_record needs a granted verdict")
    ev_by_id = {e.evidence_id: e for e in evidences}
    used = [ev_by_id[i] for i in verdict.supporting_evidence_ids
            if i in ev_by_id]
    ids = [e.evidence_id for e in used]
    c = verdict.criterion
    detail: dict = {"channels": verdict.effective_channels,
                    "action_kinds": sorted({e.action_kind for e in used})}
    if c == DETERMINISTIC_CHECK:
        detail.update({"check_id": f"get_sure:{row.utype}:{ids[0] if ids else ''}",
                       "passed": True})
    elif c == FALSIFICATION_PROBE:
        detail.update({"probe_id": f"get_sure:{ids[0] if ids else row.id}",
                       "falsified": False})
    elif c == INDEPENDENT_AUDIT:
        auditor = next((e.channel.tool_identity for e in used
                        if e.channel.method_family == MF_HUMAN), "") or actor
        if auditor == row.owner:
            raise GetSureError("independent_audit auditor equals the row owner")
        detail.update({"verdict": "confirmed"})
        return ClosureRecord(criterion=c, actor=auditor, evidence_ids=ids,
                             detail=detail, residual_u=0.0)
    elif c == EVIDENCE_SPAN:
        first = used[0] if used else None
        cite = (first.citations[0] if first and first.citations else "")
        detail.update({"span": [0, max(1, len(str(cite)))],
                       "source": str(cite) or "get_sure:evidence"})
        if not ids:
            raise GetSureError("evidence_span closure has no evidence ids")
    return ClosureRecord(criterion=c, actor=actor, evidence_ids=ids,
                         detail=detail, residual_u=0.0)


# =========================================================================
# 7. Lineage, spend, budget
# =========================================================================

def canonical_subject(row_id: str) -> str:
    """The part of a row id that names the SUBJECT rather than the run.

    `event_projections.uledger_rows_from_stream` builds ids as
    ``{correlation_id}:{kind}:{key}:{fn}#{suffix}``, and the correlation id
    changes every time the fixer re-runs. Keying lineage on the whole prefix
    (or on `correlation_group`, which embeds the same volatile base) would mint
    a FRESH lineage on every re-decision and quietly let the engine buy the
    same logical action again — the exact duplicate-spend the engine doc
    forbids. So the subject is the trailing ``path:fn`` pair.

    Two genuinely different tasks that share a path:fn inside one run collapse
    onto one lineage. That is the conservative direction: it refuses a second
    purchase rather than granting one.
    """
    parts = row_id.split(":")
    return ":".join(parts[-2:]) if len(parts) >= 2 else row_id


def doubt_lineage_id(row: URow) -> str:
    """Stable across re-decisions and across residual successors.

    Derived from canonical subject identity + doubt type + proposition class,
    NEVER from model wording and never from the run's correlation id:
    re-wording a claim, or re-running the fixer, must not mint a fresh lineage
    that can buy the same action a second time. The ledger's `~residual`
    successors deliberately fold back onto their ancestor for the same reason.
    """
    rid = row.id
    while rid.endswith("~residual"):
        rid = rid[: -len("~residual")]
    subject, _, suffix = rid.partition("#")
    return snap.cse1_digest({
        "utype": row.utype,
        "subject": canonical_subject(subject),
        "proposition": suffix.split(":")[0] if suffix else "",
    })


@dataclass(frozen=True)
class GetSureBudget:
    max_total_cost_milli: int = 100_000
    max_action_count: int = 8
    max_human_requests: int = 1
    max_wall_ms: int = 0                 # 0 == unbounded

    def __post_init__(self):
        for f in ("max_total_cost_milli", "max_action_count",
                  "max_human_requests", "max_wall_ms"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise GetSureError(f"{f} must be a non-negative int")

    def as_dict(self) -> dict:
        return {"max_total_cost_milli": self.max_total_cost_milli,
                "max_action_count": self.max_action_count,
                "max_human_requests": self.max_human_requests,
                "max_wall_ms": self.max_wall_ms}


@dataclass
class BudgetState:
    """Reserve before, settle after. A slot is never refunded — that is what
    stops a run from hiding a string of failed acquisitions."""
    budget: GetSureBudget
    spent_milli: int = 0
    actions_used: int = 0
    human_requests: int = 0
    started_ms: int = 0
    class_burn: dict = field(default_factory=dict)

    def remaining_milli(self) -> int:
        return self.budget.max_total_cost_milli - self.spent_milli

    def remaining_actions(self) -> int:
        return self.budget.max_action_count - self.actions_used

    def exhausted(self, now_ms: int | None = None) -> str:
        if self.remaining_actions() <= 0:
            return "budget_exhausted"
        if self.remaining_milli() <= 0:
            return "budget_exhausted"
        if self.budget.max_wall_ms and now_ms is not None and \
                self.started_ms and now_ms - self.started_ms >= \
                self.budget.max_wall_ms:
            return "deadline_exhausted"
        return ""

    def affordable(self, entry: ActionCatalogEntry) -> bool:
        if entry.cost_milli > self.remaining_milli():
            return False
        if entry.action_kind == ASK_HUMAN and \
                self.human_requests >= self.budget.max_human_requests:
            return False
        return True

    def settle(self, entry: ActionCatalogEntry, actual_milli: int) -> None:
        self.actions_used += 1
        self.spent_milli += max(0, int(actual_milli))
        self.class_burn[entry.cost_class] = \
            self.class_burn.get(entry.cost_class, 0) + max(0, int(actual_milli))
        if entry.action_kind == ASK_HUMAN:
            self.human_requests += 1

    def as_dict(self) -> dict:
        return {"cap": self.budget.as_dict(), "spent_milli": self.spent_milli,
                "actions_used": self.actions_used,
                "human_requests": self.human_requests,
                "remaining_milli": self.remaining_milli(),
                "remaining_actions": self.remaining_actions(),
                "class_burn": dict(sorted(self.class_burn.items()))}


class SpendLedger:
    """UNIQUE(scope, doubt_lineage_id, action_kind).

    The action TARGET is deliberately excluded: a second usage lookup for the
    same doubt is the same logical action however it is parameterised, and the
    engine refuses to buy it twice.
    """

    def __init__(self, scope_id: str):
        self.scope_id = scope_id
        self._spent: dict = {}

    def key(self, lineage_id: str, action_kind: str) -> tuple:
        return (self.scope_id, lineage_id, action_kind)

    def consumed(self, lineage_id: str, action_kind: str) -> bool:
        return self.key(lineage_id, action_kind) in self._spent

    def reserve(self, lineage_ids, action_kind: str, action_id: str) -> None:
        keys = [self.key(l, action_kind) for l in lineage_ids]
        already = [k for k in keys if k in self._spent]
        if already:
            raise GetSureError(
                f"action {action_kind} already consumed for {already[0][1]}")
        for k in keys:
            self._spent[k] = action_id

    def kinds_for(self, lineage_id: str) -> set:
        return {k[2] for k in self._spent if k[1] == lineage_id}

    def as_list(self) -> list:
        return sorted([{"lineage_id": k[1], "action_kind": k[2],
                        "action_id": v} for k, v in self._spent.items()],
                      key=lambda d: (d["lineage_id"], d["action_kind"]))


def idempotency_key(scope_id: str, lineage_ids, action_kind: str) -> str:
    return snap.cse1_digest({"scope": scope_id,
                             "lineages": sorted(set(lineage_ids)),
                             "action_kind": action_kind})


# =========================================================================
# 8. The loop
# =========================================================================

@dataclass
class GetSureContext:
    """Everything an acquisition may read. Pinned for the whole run.

    Absent fields are not silent failures: an executor that needs an absent
    field returns a `no_*` failure code naming the field, so an unfed engine is
    distinguishable from a broken one.
    """
    base_facts: dict = field(default_factory=dict)
    usage_index: object = None
    knowledge_ledger: object = None
    witness_inputs: dict = field(default_factory=dict)
    corpus: tuple = ()
    seam_graph: dict = field(default_factory=dict)
    human_answers: dict = field(default_factory=dict)
    repro_pairs: tuple = ()              # consumed by the repro_flip seam
    probe_traces: tuple = ()             # consumed by the instrumentation seam
    environment_id: str = "hermetic"

    def digest(self) -> str:
        return snap.cse1_digest({
            "base_facts": _snapshot_safe(self.base_facts),
            "has_usage_index": self.usage_index is not None,
            "witness_inputs": sorted(self.witness_inputs or {}),
            "corpus": len(self.corpus),
            "seam_graph": sorted(self.seam_graph or {}),
            "human_answers": sorted(self.human_answers or {}),
            "environment_id": self.environment_id})


@dataclass(frozen=True)
class ReDecideRequest:
    """What `decide` receives. Note what it CANNOT receive: the previous
    verdict, the previous ledger, or any 'now close row 4' instruction. Fresh
    derivation is enforced by the absence of the field, not by a convention."""
    run_id: str
    context: GetSureContext
    accepted_evidence: tuple
    action_spend: tuple
    budget_state: dict
    catalog_digest: str
    tier: str
    iteration: int


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    parent_id: str | None
    ledger: ULedger
    context_digest: str
    iteration: int
    created_at_micros: int


@dataclass(frozen=True)
class ActionRequest:
    action_id: str
    run_id: str
    source_decision_id: str
    target_doubt_lineage_ids: tuple
    action_kind: str
    target: dict
    context_digest: str
    catalog_digest: str
    cost_estimate_milli: int
    idempotency_key: str


@dataclass(frozen=True)
class SureResult:
    status: str
    run_id: str
    decision_id: str
    context_digest: str
    tier: str
    retired_lineage_ids: tuple
    evidence_ids: tuple
    closure_checks: tuple
    action_count: int
    actual_cost_milli: int
    open_doubt_count: int
    unretired_row_ids: tuple
    proof_digest: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "run_id": self.run_id,
                "decision_id": self.decision_id,
                "context_digest": self.context_digest, "tier": self.tier,
                "retired_lineage_ids": list(self.retired_lineage_ids),
                "evidence_ids": list(self.evidence_ids),
                "closure_checks": [dict(c) for c in self.closure_checks],
                "action_count": self.action_count,
                "actual_cost_milli": self.actual_cost_milli,
                "open_doubt_count": self.open_doubt_count,
                "unretired_row_ids": list(self.unretired_row_ids),
                "proof_digest": self.proof_digest}


@dataclass(frozen=True)
class PricedAbstention:
    status: str
    run_id: str
    decision_id: str
    context_digest: str
    tier: str
    reason: str
    budget: dict
    remaining_doubts: tuple
    next_best_action: dict | None
    lower_bound_additional_cost_milli: int
    evidence_ids: tuple
    proof_digest: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "run_id": self.run_id,
                "decision_id": self.decision_id,
                "context_digest": self.context_digest, "tier": self.tier,
                "reason": self.reason, "budget": self.budget,
                "remaining_doubts": [dict(d) for d in self.remaining_doubts],
                "next_best_action": self.next_best_action,
                "lower_bound_additional_cost_milli":
                    self.lower_bound_additional_cost_milli,
                "evidence_ids": list(self.evidence_ids),
                "proof_digest": self.proof_digest}


@dataclass(frozen=True)
class ContradictionEscalation:
    status: str
    run_id: str
    decision_id: str
    contradiction_evidence_ids: tuple
    conflicting_predicates: tuple
    affected_lineage_ids: tuple
    action_history: tuple
    required_human_questions: tuple
    safe_next_steps: tuple
    proof_digest: str = ""

    def as_dict(self) -> dict:
        return {"status": self.status, "run_id": self.run_id,
                "decision_id": self.decision_id,
                "contradiction_evidence_ids":
                    list(self.contradiction_evidence_ids),
                "conflicting_predicates": list(self.conflicting_predicates),
                "affected_lineage_ids": list(self.affected_lineage_ids),
                "action_history": [dict(a) for a in self.action_history],
                "required_human_questions":
                    list(self.required_human_questions),
                "safe_next_steps": list(self.safe_next_steps),
                "proof_digest": self.proof_digest}


def verify_result(request: ActionRequest, entry: ActionCatalogEntry,
                  result: ExecutorResult, *, run_id: str,
                  context_digest: str, now_micros: int) -> EvidenceRecord:
    """Executor claim -> verified evidence. DOWNGRADE ONLY.

    An adapter may say `support`; the verifier may say `inconclusive`. It may
    never go the other way, and it may never raise a claimed strength.
    """
    channel = result.channel or Channel(
        channel_id=f"{entry.action_kind}:{request.action_id}",
        method_family=entry.method_family)
    if channel.method_family != entry.method_family:
        # A channel that mislabels its own family is invalid, not creative.
        assessment = INVALID
    elif result.failure_code:
        assessment = INVALID if result.claimed_assessment == CONTRADICTION \
            else INCONCLUSIVE
    else:
        assessment = result.claimed_assessment
    if result.claimed_assessment == CONTRADICTION and not result.failure_code:
        assessment = CONTRADICTION
    if assessment not in ASSESSMENTS:
        assessment = INVALID
    strength = result.claimed_strength if assessment == SUPPORT else WEAK
    ev_id = snap.cse1_digest({
        "action_id": request.action_id, "kind": entry.action_kind,
        "obs": _snapshot_safe(list(result.observations)),
        "assessment": assessment})
    return EvidenceRecord(
        evidence_id=ev_id, action_id=request.action_id, run_id=run_id,
        source_decision_id=request.source_decision_id,
        doubt_lineage_ids=request.target_doubt_lineage_ids,
        action_kind=entry.action_kind, channel=channel, strength=strength,
        assessment=assessment, observations=result.observations,
        citations=result.citations, coverage=result.coverage,
        margin=result.margin, truncation=result.truncation,
        detail=dict(result.detail or {}, failure_code=result.failure_code),
        context_digest_before=context_digest, created_at_micros=now_micros)


def _blocking_rows(ledger: ULedger, policy) -> list:
    ready, blocking = ledger.ready_to_act(policy)
    ids = list(blocking)
    return [ledger.rows[i] for i in ids if i in ledger.rows]


def _select_action(ledger: ULedger, policy, spend: SpendLedger,
                   budget: BudgetState, catalog: dict):
    """Revalidate the ledger's own recommendation, then fall back.

    The ledger already computes `next_action` (cheapest-biggest-dU). The engine
    EXECUTES it rather than re-deriving a preference — but it must still be
    revalidated against the pinned catalog, the spend ledger and the remaining
    budget, because the recommendation was computed without any of them.
    """
    tried = []
    rec = ledger.next_action(policy)
    ordered = []
    if rec is not None:
        ordered.append((rec[0], rec[1].action, rec[1]))
    blocking = _blocking_rows(ledger, policy)
    for row in blocking:
        for a in row.reduction_actions:
            ordered.append((row.id, a.action, a))
    for rid, name, action in ordered:
        row = ledger.rows.get(rid)
        if row is None or not row.open:
            continue
        kind = action_for_name(name)
        if kind == UNMAPPED:
            tried.append({"row_id": rid, "action": name,
                          "skip_reason": "unmapped_action_name"})
            continue
        entry = catalog.get(kind)
        if entry is None:
            tried.append({"row_id": rid, "action": name,
                          "skip_reason": "not_in_catalog"})
            continue
        if not entry.implemented:
            tried.append({"row_id": rid, "action": name, "action_kind": kind,
                          "skip_reason": "not_implemented", "seam": entry.seam,
                          "cost_milli": entry.cost_milli})
            continue
        if not entry.eligible_for(row.utype):
            tried.append({"row_id": rid, "action": name, "action_kind": kind,
                          "skip_reason": f"forbidden_for_{row.utype}"})
            continue
        lineage = doubt_lineage_id(row)
        if spend.consumed(lineage, kind):
            tried.append({"row_id": rid, "action": name, "action_kind": kind,
                          "skip_reason": "already_consumed"})
            continue
        if not budget.affordable(entry):
            tried.append({"row_id": rid, "action": name, "action_kind": kind,
                          "skip_reason": "unaffordable",
                          "cost_milli": entry.cost_milli})
            continue
        if row.closure_criterion and row.closure_criterion not in \
                entry.produces_criteria:
            tried.append({"row_id": rid, "action": name, "action_kind": kind,
                          "skip_reason":
                              f"cannot_produce_{row.closure_criterion}"})
            continue
        return row, entry, lineage, tried
    return None, None, None, tried


def _target_for(row: URow, ctx: GetSureContext) -> dict:
    """Canonical target for an action against `row`. The subject comes from the
    row id (`...:<path>:<fn>#<suffix>`), never from the claim prose."""
    rid = row.id.split("#")[0]
    parts = rid.split(":")
    fn = parts[-1] if parts else ""
    path = parts[-2] if len(parts) >= 2 else ""
    return {"row_id": row.id, "subject": f"{path}:{fn}" if path else rid,
            "symbol": fn, "path": path,
            "seam": row.correlation_group or rid,
            "question": row.claim[:400],
            "question_id": f"q:{doubt_lineage_id(row)[:20]}"}


def run_get_sure(*, run_id: str, ledger: ULedger, context: GetSureContext,
                 decide=None, budget: GetSureBudget | None = None,
                 tier: str = R1, catalog: dict | None = None,
                 recorder: "MeasurementRecorder | None" = None,
                 now_ms=None, dry_run: bool = False):
    """Spend evidence budget against `ledger` until SURE / priced / escalated.

    `decide` is INJECTED: `decide(ReDecideRequest) -> ULedger`. Modelless tests
    pass a stub; the live wiring is documented in the module seam notes. When
    `decide` is None the engine re-uses the current ledger object as the fresh
    derivation, which is honest only because every closure it applies went
    through `check_closure` first — it is the degenerate case, not the design.
    """
    catalog = catalog or CATALOG
    budget = budget or GetSureBudget()
    entry = RISK_TIERS.get(tier)
    if entry is None:
        raise GetSureError(f"unknown risk tier {tier!r}")
    policy = policy_for_tier(tier)
    clock = now_ms or (lambda: int(time.time() * 1000))
    state = BudgetState(budget=budget, started_ms=clock())
    spend = SpendLedger(run_id)
    cat_digest = catalog_digest(catalog)

    decisions: list = []
    evidences: list = []
    closure_checks: list = []
    action_history: list = []
    retired_lineages: list = []
    iteration = 0

    def _record_decision(led, parent):
        nonlocal iteration
        d = DecisionRecord(
            decision_id=snap.cse1_digest(
                {"run": run_id, "iter": iteration,
                 "rows": sorted(r.id for r in led.rows.values() if r.open)}),
            parent_id=parent, ledger=led, context_digest=context.digest(),
            iteration=iteration, created_at_micros=int(time.time() * 1_000_000))
        decisions.append(d)
        return d

    current = _record_decision(ledger, None)

    while True:
        led = current.ledger
        ready, blocking_ids = led.ready_to_act(policy)
        open_rows = [r for r in led.rows.values() if r.open]

        if ready:
            result = SureResult(
                status=SURE, run_id=run_id, decision_id=current.decision_id,
                context_digest=current.context_digest, tier=tier,
                retired_lineage_ids=tuple(dict.fromkeys(retired_lineages)),
                evidence_ids=tuple(e.evidence_id for e in evidences),
                closure_checks=tuple(closure_checks),
                action_count=state.actions_used,
                actual_cost_milli=state.spent_milli,
                open_doubt_count=len(open_rows),
                unretired_row_ids=tuple(sorted(r.id for r in open_rows)))
            result = replace(result, proof_digest=snap.cse1_digest(
                _snapshot_safe(result.as_dict())))
            if recorder:
                recorder.finish(run_id, result)
            return result, decisions, evidences

        stop = state.exhausted(clock())
        if stop:
            return _price(run_id, current, tier, state, led, policy, spend,
                          catalog, evidences, stop, recorder, decisions)

        row, cat_entry, lineage, tried = _select_action(
            led, policy, spend, state, catalog)
        if row is None:
            return _price(run_id, current, tier, state, led, policy, spend,
                          catalog, evidences, "no_eligible_action", recorder,
                          decisions, tried=tried)

        target = _target_for(row, context)
        action_id = snap.cse1_digest(
            {"run": run_id, "lineage": lineage, "kind": cat_entry.action_kind})
        request = ActionRequest(
            action_id=action_id, run_id=run_id,
            source_decision_id=current.decision_id,
            target_doubt_lineage_ids=(lineage,),
            action_kind=cat_entry.action_kind, target=target,
            context_digest=current.context_digest, catalog_digest=cat_digest,
            cost_estimate_milli=cat_entry.cost_milli,
            idempotency_key=idempotency_key(run_id, (lineage,),
                                            cat_entry.action_kind))

        if dry_run:
            action_history.append({"action_id": action_id, "row_id": row.id,
                                   "action_kind": cat_entry.action_kind,
                                   "cost_milli": cat_entry.cost_milli,
                                   "executed": False})
            spend.reserve((lineage,), cat_entry.action_kind, action_id)
            state.settle(cat_entry, 0)
            continue

        spend.reserve((lineage,), cat_entry.action_kind, action_id)
        started = clock()
        try:
            exec_result = cat_entry.executor(request, context)
        except Exception as exc:                  # an adapter crash is a cost
            exec_result = ExecutorResult(
                failure_code=f"adapter_error:{type(exc).__name__}",
                cost_milli=cat_entry.cost_milli)
        actual = exec_result.cost_milli or cat_entry.cost_milli
        state.settle(cat_entry, actual)

        ev = verify_result(request, cat_entry, exec_result, run_id=run_id,
                           context_digest=current.context_digest,
                           now_micros=int(time.time() * 1_000_000))
        evidences.append(ev)
        action_history.append({"action_id": action_id, "row_id": row.id,
                               "action_kind": cat_entry.action_kind,
                               "cost_milli": actual, "executed": True,
                               "assessment": ev.assessment,
                               "wall_ms": max(0, clock() - started)})
        if recorder:
            recorder.action(run_id=run_id, row=row, lineage_id=lineage,
                            request=request, evidence=ev, cost_milli=actual,
                            wall_ms=max(0, clock() - started))

        if ev.assessment == CONTRADICTION:
            esc = ContradictionEscalation(
                status=ESCALATE_CONTRADICTION, run_id=run_id,
                decision_id=current.decision_id,
                contradiction_evidence_ids=(ev.evidence_id,),
                conflicting_predicates=tuple(
                    f"{o.get('kind', 'observation')}={o}"
                    for o in ev.observations),
                affected_lineage_ids=(lineage,),
                action_history=tuple(action_history),
                required_human_questions=(target["question"],),
                safe_next_steps=("freeze automatic acquisition",
                                 "adjudicate both evidence chains",
                                 "re-decide only after adjudication"))
            esc = replace(esc, proof_digest=snap.cse1_digest(
                _snapshot_safe(esc.as_dict())))
            if recorder:
                recorder.finish(run_id, esc)
            return esc, decisions, evidences

        # Closure is attempted ONLY through the code-owned predicate.
        row_ev = [e for e in evidences if lineage in e.doubt_lineage_ids]
        verdict = check_closure(row, row_ev, tier, catalog=catalog)
        closure_checks.append({"row_id": row.id, "lineage_id": lineage,
                               **verdict.as_dict()})
        if recorder:
            recorder.resolution(run_id=run_id, row=row, lineage_id=lineage,
                                verdict=verdict, action_id=action_id)
        if verdict.retire:
            try:
                led.close(row.id, to_closure_record(row, verdict, row_ev))
                retired_lineages.append(lineage)
            except (ClosureRejected, GetSureError) as exc:
                # The ledger's own gate refused. That is a REAL outcome, not a
                # reason to weaken either gate.
                closure_checks[-1]["ledger_rejected"] = f"{type(exc).__name__}: {exc}"

        iteration += 1
        if decide is not None:
            fresh = decide(ReDecideRequest(
                run_id=run_id, context=context,
                accepted_evidence=tuple(evidences),
                action_spend=tuple(spend.as_list()),
                budget_state=state.as_dict(), catalog_digest=cat_digest,
                tier=tier, iteration=iteration))
            if not isinstance(fresh, ULedger):
                raise GetSureError("decide must return a ULedger")
            current = _record_decision(fresh, current.decision_id)
        else:
            current = _record_decision(led, current.decision_id)


def _price(run_id, current, tier, state, led, policy, spend, catalog,
           evidences, reason, recorder, decisions, tried=()):
    """Say exactly what remained unresolved. 'Insufficient confidence' is not
    an abstention reason (engine doc 8.2)."""
    remaining = []
    lower_bound = 0
    for row in sorted((r for r in led.rows.values() if r.open),
                      key=lambda r: r.id):
        lineage = doubt_lineage_id(row)
        consumed = sorted(spend.kinds_for(lineage))
        options = []
        for a in row.reduction_actions:
            kind = action_for_name(a.action)
            e = catalog.get(kind)
            options.append({
                "action": a.action, "action_kind": kind,
                "cost_milli": e.cost_milli if e else None,
                "implemented": bool(e and e.implemented),
                "seam": (e.seam if e and not e.implemented else ""),
                "consumed": kind in consumed,
                "can_retire": bool(e and e.can_retire(row.utype))})
        live = [o for o in options if o["implemented"] and not o["consumed"]
                and o["can_retire"]]
        cheapest = min(live, key=lambda o: o["cost_milli"]) if live else None
        if cheapest:
            lower_bound += int(cheapest["cost_milli"])
        remaining.append({
            "row_id": row.id, "lineage_id": lineage, "type": row.utype,
            "claim": row.claim[:200], "u_raw_milli": int(round(row.u * 1000)),
            "severity_milli": int(round(row.severity * 1000)),
            "hard": row.irreversibility > 0.0,
            "closure_criterion": row.closure_criterion,
            "actions_already_consumed": consumed,
            "options": options,
            "cheapest_remaining_action": cheapest})
    ranked = [d["cheapest_remaining_action"] for d in remaining
              if d["cheapest_remaining_action"]]
    nxt = min(ranked, key=lambda o: o["cost_milli"]) if ranked else None
    out = PricedAbstention(
        status=PRICED_ABSTENTION, run_id=run_id,
        decision_id=current.decision_id, context_digest=current.context_digest,
        tier=tier, reason=reason, budget=state.as_dict(),
        remaining_doubts=tuple(remaining), next_best_action=nxt,
        lower_bound_additional_cost_milli=lower_bound,
        evidence_ids=tuple(e.evidence_id for e in evidences))
    out = replace(out, proof_digest=snap.cse1_digest(
        _snapshot_safe(out.as_dict())))
    if recorder:
        recorder.finish(run_id, out, skipped=list(tried))
    return out, decisions, evidences


# =========================================================================
# 9. Measurement (a_getsure_measure.md)
# =========================================================================

@dataclass
class MeasurementRecorder:
    """Emits exactly the rows the w63 measurement design needs.

    Two retirement numbers are kept apart on purpose: `operational` is what the
    loop said, `verified` is what a blinded evaluator (or gold) said. Reporting
    only the first is how a broken resolver looks successful.
    """
    abstention_events: list = field(default_factory=list)
    doubt_rows: list = field(default_factory=list)
    action_attempts: list = field(default_factory=list)
    resolutions: list = field(default_factory=list)
    final_verdicts: list = field(default_factory=list)

    def open_run(self, run_id: str, task_id: str, ledger: ULedger, tier: str):
        rows = [r for r in ledger.rows.values() if r.open]
        self.abstention_events.append({
            "abstention_event_id": f"{run_id}:initial", "task_id": task_id,
            "run_id": run_id, "stage": "initial", "initial_verdict": "abstain",
            "tier": tier, "n_doubts": len(rows),
            "loop_eligible": bool(rows),
            "loop_skip_reason": "" if rows else "no_doubt_rows"})
        for r in rows:
            self.doubt_rows.append({
                "doubt_id": r.id, "run_id": run_id, "task_id": task_id,
                "lineage_id": doubt_lineage_id(r), "doubt_type": r.utype,
                "doubt_statement": r.claim[:400],
                "severity_milli": int(round(r.severity * 1000)),
                "u_raw_milli": int(round(r.u * 1000)),
                "eligible_action_classes": sorted(
                    {action_for_name(a.action) for a in r.reduction_actions}
                    - {UNMAPPED})})
        if not rows:
            # Never silently drop an unclassified abstention from the
            # denominator (measure doc section 2).
            self.doubt_rows.append({
                "doubt_id": f"{run_id}:UNCLASSIFIED", "run_id": run_id,
                "task_id": task_id, "lineage_id": "",
                "doubt_type": "UNCLASSIFIED_MISSING_REASON",
                "doubt_statement": "system abstained without a structured "
                                   "reason", "severity_milli": 1000,
                "u_raw_milli": 1000, "eligible_action_classes": []})

    def action(self, *, run_id, row, lineage_id, request, evidence,
               cost_milli, wall_ms):
        self.action_attempts.append({
            "action_id": request.action_id, "run_id": run_id,
            "doubt_id": row.id, "lineage_id": lineage_id,
            "doubt_type": row.utype, "action_class": request.action_kind,
            "sequence_index": len(self.action_attempts), "selected": True,
            "evidence_id": evidence.evidence_id,
            "assessment": evidence.assessment,
            "correlation_class": evidence.channel.correlation_class(),
            "cost_milli": int(cost_milli), "wall_ms": int(wall_ms),
            "failure_code": evidence.detail.get("failure_code", "")})

    def resolution(self, *, run_id, row, lineage_id, verdict, action_id):
        self.resolutions.append({
            "run_id": run_id, "doubt_id": row.id, "lineage_id": lineage_id,
            "doubt_type": row.utype,
            "resolution_status": "retired" if verdict.retire else "unresolved",
            "retirement_source": action_id if verdict.retire else "",
            "retirement_reason": "; ".join(verdict.reasons)[:400],
            "effective_channels": verdict.effective_channels,
            "loop_self_assessment": bool(verdict.retire),
            "independent_assessment": None})

    def finish(self, run_id: str, terminal, skipped=()):
        status = getattr(terminal, "status", "")
        flipped = status == SURE
        self.final_verdicts.append({
            "run_id": run_id, "initial_verdict": "abstain",
            "final_verdict": "answer" if flipped else "abstain",
            "flipped": flipped,
            "flip_type": "abstain_to_answer" if flipped else "no_flip",
            "terminal_reason": getattr(terminal, "reason", status),
            "status": status,
            "open_doubt_count": getattr(terminal, "open_doubt_count", None),
            "unresolved_blocking": list(
                getattr(terminal, "unretired_row_ids", ()) or ()),
            "action_count": getattr(terminal, "action_count",
                                    len(self.action_attempts)),
            "actual_cost_milli": getattr(terminal, "actual_cost_milli",
                                         (getattr(terminal, "budget", {}) or {})
                                         .get("spent_milli", 0)),
            "skipped_actions": [dict(s) for s in skipped]})

    def as_dict(self) -> dict:
        return {"abstention_events": list(self.abstention_events),
                "doubt_rows": list(self.doubt_rows),
                "action_attempts": list(self.action_attempts),
                "resolutions": list(self.resolutions),
                "final_verdicts": list(self.final_verdicts)}


def one_sided_upper_bound_95(k: int, n: int) -> float:
    """Wilson-score one-sided 95% upper bound for k/n.

    A point estimate below the base rate is not enough (measure doc 6): the
    safety gate is on the BOUND. With k == 0 this reproduces the rule-of-three
    behaviour (n ~ 3/tau) the doc plans capacity with.
    """
    if not isinstance(n, int) or n < 0 or not isinstance(k, int) or k < 0:
        raise GetSureError("k and n must be non-negative ints")
    if n == 0:
        return 1.0
    if k > n:
        raise GetSureError("k cannot exceed n")
    z = 1.6448536269514722
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / denom)


def measurement_report(recorder: MeasurementRecorder, gold: dict | None = None,
                       *, base_wrong_rate: float | None = None) -> dict:
    """Doubt-type and global metrics from recorded rows.

    `gold` maps run_id -> {"gold_decision": "answer"|"abstain",
                           "answer_correct": bool}. Without gold, the false-sure
    columns are reported as `not_measured` rather than as zero — an unmeasured
    safety metric that prints 0.0 is exactly the false comfort this programme
    exists to prevent.
    """
    gold = gold or {}
    by_type: dict = {}
    for d in recorder.doubt_rows:
        b = by_type.setdefault(d["doubt_type"], {
            "D": 0, "R_operational": 0, "R_verified": 0, "F": 0, "C": 0,
            "W": 0, "P": 0, "tokens_milli_cost": 0, "wall_ms": 0})
        b["D"] += 1
    for r in recorder.resolutions:
        b = by_type.setdefault(r["doubt_type"], {
            "D": 0, "R_operational": 0, "R_verified": 0, "F": 0, "C": 0,
            "W": 0, "P": 0, "tokens_milli_cost": 0, "wall_ms": 0})
        if r["resolution_status"] == "retired":
            b["R_operational"] += 1
            if r.get("independent_assessment") is True:
                b["R_verified"] += 1
    for a in recorder.action_attempts:
        b = by_type.setdefault(a["doubt_type"], {
            "D": 0, "R_operational": 0, "R_verified": 0, "F": 0, "C": 0,
            "W": 0, "P": 0, "tokens_milli_cost": 0, "wall_ms": 0})
        b["tokens_milli_cost"] += a["cost_milli"]
        b["wall_ms"] += a["wall_ms"]

    run_types: dict = {}
    for d in recorder.doubt_rows:
        run_types.setdefault(d["run_id"], set()).add(d["doubt_type"])

    flips = wrong = correct = preserved = 0
    unsafe_flips = []
    for v in recorder.final_verdicts:
        g = gold.get(v["run_id"])
        types = run_types.get(v["run_id"], set())
        if v["flipped"]:
            flips += 1
            if v.get("unresolved_blocking"):
                # Protocol violation: answered with a blocking doubt open.
                unsafe_flips.append(v["run_id"])
            if g is not None:
                ok = (g.get("gold_decision") == "answer"
                      and bool(g.get("answer_correct")))
                correct += 1 if ok else 0
                wrong += 0 if ok else 1
                for t in types:
                    by_type[t]["F"] += 1
                    by_type[t]["C" if ok else "W"] += 1
            else:
                for t in types:
                    by_type[t]["F"] += 1
        else:
            preserved += 1
            for t in types:
                by_type[t]["P"] += 1

    D_total = len({d["run_id"] for d in recorder.doubt_rows})
    measured = bool(gold)
    out_types = {}
    for t, b in sorted(by_type.items()):
        D = b["D"] or 0
        out_types[t] = {
            **b,
            "retirement_rate": _ratio(b["R_operational"], D),
            "retirement_precision": (
                _ratio(b["R_verified"], b["R_operational"])
                if b["R_operational"] and any(
                    r.get("independent_assessment") is not None
                    for r in recorder.resolutions) else "not_measured"),
            "flip_rate_given_retirement": _ratio(b["F"], b["R_operational"]),
            "correct_flip_rate": _ratio(b["C"], b["F"]) if measured
                else "not_measured",
            "correct_conversion_rate": _ratio(b["C"], D) if measured
                else "not_measured",
            "false_sure_rate": _ratio(b["W"], D) if measured
                else "not_measured",
            "false_sure_rate_ub95": (one_sided_upper_bound_95(b["W"], D)
                                     if measured and D else "not_measured"),
            "abstention_preservation": _ratio(b["P"], D),
            "cost_milli_per_correct_conversion": (
                _ratio(b["tokens_milli_cost"], b["C"]) if b["C"]
                else "not_achieved"),
        }

    global_fs = _ratio(wrong, D_total) if measured else "not_measured"
    ub = one_sided_upper_bound_95(wrong, D_total) if measured and D_total \
        else "not_measured"
    gate = "not_measured"
    if measured and base_wrong_rate is not None and D_total:
        gate = "pass" if ub < base_wrong_rate else "fail"
    return {
        "units": {"initial_abstentions": D_total,
                  "doubt_rows": len(recorder.doubt_rows),
                  "action_attempts": len(recorder.action_attempts)},
        "by_doubt_type": out_types,
        "global": {
            "flips": flips, "correct_conversions": correct if measured else
                "not_measured",
            "wrong_conversions": wrong if measured else "not_measured",
            "preserved_abstentions": preserved,
            "false_sure_rate": global_fs,
            "false_sure_rate_ub95": ub,
            "base_wrong_rate": base_wrong_rate
            if base_wrong_rate is not None else "not_supplied",
            "safety_gate": gate,
            "protocol_violations": unsafe_flips,
        },
    }


def _ratio(num, den):
    return round(num / den, 6) if den else "not_achieved"


# =========================================================================
# 10. Replay dry-run over real cycle rows
# =========================================================================

_SUFFIX_TO_UTYPE = {
    "abstain": UNKNOWN_ATOM,
    "unverified": UNVERIFIED_CLAIM,
    "branch": UNEXPLORED_BRANCH,
    "margin": THIN_MARGIN,
}


def _utype_from_row_id(row_id: str) -> str | None:
    _, _, suffix = row_id.partition("#")
    return _SUFFIX_TO_UTYPE.get(suffix.split(":")[0]) if suffix else None


def replay_cycle_uledger(path: str, *, catalog: dict | None = None) -> dict:
    """Dry-run the loop's SELECTION over real recorded cycles.

    This runs no acquisitions. It answers exactly two questions per cycle:
    which catalog action would fire against the recorded `next_action`, and
    what it would cost.

    HONESTY NOTE that the caller must not lose: `blocking_rows` in this file is
    written truncated to the first 10 while `n_blocking` is the true count, so
    the per-row breakdown below covers `rows_listed`, not every blocking row.
    Both numbers are reported; a table that quoted only the first would be a
    completeness claim made through a truncating view.
    """
    import json
    catalog = catalog or CATALOG
    cycles = []
    listed = truncated_rows = 0
    by_kind: dict = {}
    by_utype: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cycles.append(json.loads(line))

    rows_out = []
    for c in cycles:
        n_block = int(c.get("n_blocking", 0) or 0)
        blocking = list(c.get("blocking_rows") or [])
        listed += len(blocking)
        truncated_rows += max(0, n_block - len(blocking))
        for rid in blocking:
            ut = _utype_from_row_id(rid)
            if ut:
                by_utype[ut] = by_utype.get(ut, 0) + 1
        na = c.get("next_action")
        if not na:
            rows_out.append({"cycle_id": c.get("cycle_id"),
                             "ready_to_act": bool(c.get("ready_to_act")),
                             "n_blocking": n_block, "action": None,
                             "action_kind": None, "would_fire": False,
                             "reason": "ledger recommended no action",
                             "cost_milli": 0})
            continue
        kind = action_for_name(na.get("action", ""))
        entry = catalog.get(kind)
        ut = _utype_from_row_id(na.get("row_id", "") or "")
        if kind == UNMAPPED:
            reason = "unmapped: re-asking a model is not an evidence channel"
            fire, cost = False, 0
        elif entry is None:
            reason = "no catalog entry"
            fire, cost = False, 0
        elif not entry.implemented:
            reason = f"stubbed: {entry.seam[:60]}..."
            fire, cost = False, 0
        elif ut and not entry.eligible_for(ut):
            reason = f"forbidden for {ut}"
            fire, cost = False, 0
        elif ut and not entry.can_retire(ut):
            reason = f"contributes to {ut} but cannot retire it alone"
            fire, cost = True, entry.cost_milli
        else:
            reason = "would execute"
            fire, cost = True, entry.cost_milli
        if entry is not None:
            b = by_kind.setdefault(entry.action_kind,
                                   {"selected": 0, "would_fire": 0,
                                    "cost_milli": 0,
                                    "implemented": entry.implemented})
            b["selected"] += 1
            b["would_fire"] += int(fire)
            b["cost_milli"] += cost
        else:
            b = by_kind.setdefault(UNMAPPED, {"selected": 0, "would_fire": 0,
                                              "cost_milli": 0,
                                              "implemented": False})
            b["selected"] += 1
        rows_out.append({
            "cycle_id": c.get("cycle_id"),
            "ready_to_act": bool(c.get("ready_to_act")),
            "n_blocking": n_block, "action": na.get("action"),
            "row_utype": ut, "action_kind": None if kind == UNMAPPED else kind,
            "would_fire": fire, "reason": reason, "cost_milli": cost})

    return {
        "path": path,
        "cycles": len(cycles),
        "cycles_ready": sum(1 for c in cycles if c.get("ready_to_act")),
        "cycles_with_recommendation": sum(1 for c in cycles
                                          if c.get("next_action")),
        "blocking_rows_total": sum(int(c.get("n_blocking", 0) or 0)
                                   for c in cycles),
        "blocking_rows_listed": listed,
        "blocking_rows_truncated_away": truncated_rows,
        "blocking_by_utype_listed_only": dict(sorted(by_utype.items())),
        "by_action_kind": {k: by_kind[k] for k in sorted(by_kind)},
        "would_fire": sum(1 for r in rows_out if r["would_fire"]),
        "would_not_fire": sum(1 for r in rows_out if not r["would_fire"]),
        "total_cost_milli": sum(r["cost_milli"] for r in rows_out),
        "rows": rows_out,
    }


def format_replay_table(report: dict) -> str:
    lines = [
        f"replay: {report['path']}",
        f"  cycles={report['cycles']} ready={report['cycles_ready']} "
        f"with_recommendation={report['cycles_with_recommendation']}",
        f"  blocking rows: total={report['blocking_rows_total']} "
        f"listed={report['blocking_rows_listed']} "
        f"truncated_away={report['blocking_rows_truncated_away']} "
        f"(per-row breakdown covers LISTED only)",
        f"  blocking by type (listed): "
        f"{report['blocking_by_utype_listed_only']}",
        "",
        f"  {'action_kind':<32}{'sel':>5}{'fire':>6}{'impl':>7}{'cost_milli':>12}",
    ]
    for kind, b in report["by_action_kind"].items():
        lines.append(f"  {kind:<32}{b['selected']:>5}{b['would_fire']:>6}"
                     f"{str(b['implemented']):>7}{b['cost_milli']:>12}")
    lines.append("")
    lines.append(f"  would_fire={report['would_fire']} "
                 f"would_not_fire={report['would_not_fire']} "
                 f"total_cost_milli={report['total_cost_milli']}")
    return "\n".join(lines)
