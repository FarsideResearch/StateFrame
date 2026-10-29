"""ATTENTION ALLOCATOR — where a fixed compute budget gets spent.

Design of record: `sol_answers/w8/a_attention.md` (floor-constrained,
risk-prioritized water-filling), plus the crucible correction that the design
does not state loudly enough and which this module exists to make impossible:

    A CONFIDENTLY-WRONG UPSTREAM PICK STARVES DOWNSTREAM VERIFICATION.

Uncertainty buys depth. That is the whole point of the allocator, and it is
also its failure mode: if the only signal is "how unsure am I", then a stage
that is *sure and wrong* asks for nothing, gets nothing, and the verification
that would have caught it is defunded by its own victim's confidence. Margin is
a fragility probe, not a correctness oracle. So:

    FLOORS ARE ALLOCATED FIRST, UNCONDITIONALLY, AND NEVER SCALE WITH
    CONFIDENCE. A max-margin-everywhere task still buys floor witness inputs
    and floor deliberation branches. If the floors do not fit in the budget the
    task fails CLOSED with ESCALATE — verification is never shaved to make the
    arithmetic work.

Everything here is a pure function of its arguments: no clock, no I/O, no
model calls, no randomness. Two calls with equal inputs return equal output,
which is what makes an allocation replayable from the event stream.

Shape (per the design):

    A_s = S_s x U_s x K_s              stake x uncertainty x consequence
    M_s(k) = (A_s / P_s) / (1 + k/tau_s)   diminishing returns past the floor

Floors go out first; the post-floor remainder minus a preemption reserve is
water-filled one atomic unit at a time to the highest marginal score, with
deterministic tie-breaking by stage order.

Re-planning at a checkpoint is hysteretic: a challenger must beat the standing
plan by `hysteresis_ratio` on `consecutive_required` checkpoints in a row
before the plan moves. Oscillating uncertainty therefore does not oscillate the
allocation. A declared safety violation bypasses hysteresis and immediately
funds the verification closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .decision_event import (
    ESCALATE,
    PLAN_REGISTERED,
    Producer,
    content_ref,
)

MODULE_VERSION = "1"

# --------------------------------------------------------------------- stages
# The stage name IS the depth key. One spelling, everywhere (decision_event.py
# doctrine #1): there is no translation table to drift out of sync.
RETRIEVAL = "retrieval_width"
MENU = "menu_retries"
BALLOT = "ballot_budget"
WITNESS = "witness_inputs"
DELIBERATION = "deliberation_branches"

STAGES: tuple[str, ...] = (RETRIEVAL, MENU, BALLOT, WITNESS, DELIBERATION)
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

#: Stages whose whole job is catching an upstream mistake. Their floors are the
#: crucible fix; their stake weight is what a high stakes tier buys more of.
VERIFICATION_STAGES = frozenset({WITNESS, DELIBERATION})

STAKES_TIERS: tuple[str, ...] = ("low", "medium", "high", "critical")


class AttentionError(ValueError):
    """Malformed allocator input. Raised at the boundary, never swallowed."""


# --------------------------------------------------------------------- policy
@dataclass(frozen=True)
class StagePolicy:
    """Per-stage constants. `floor` is a safety property, not a preference."""
    floor: int
    cap: int
    price: float          # P_s: compute price per depth unit beyond zero
    stake: float          # S_s base: U-ledger policy stake for this stage
    consequence: float    # K_s: repair cost / blast radius if this stage is wrong
    tau: float            # diminishing-returns constant

    def __post_init__(self):
        if self.floor < 0:
            raise AttentionError(f"floor must be >= 0, got {self.floor}")
        if self.cap < self.floor:
            raise AttentionError(f"cap {self.cap} below floor {self.floor}")
        for name in ("price", "stake", "consequence", "tau"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                raise AttentionError(f"{name} must be a positive number, got {v!r}")


#: Defaults. The four floors come straight from the design's Q3 commit
#: (retrieval 2 independent paths, 1 menu retry beyond the first attempt,
#: 2 independent witness inputs, 2 deliberation branches including a skeptic).
#: `ballot_budget` is this codebase's line-ballot rung, which the design predates;
#: its floor of 1 keeps one independent line vote alive on every task.
DEFAULT_POLICIES: Mapping[str, StagePolicy] = {
    RETRIEVAL:    StagePolicy(floor=2, cap=48, price=1.0, stake=1.0, consequence=0.8, tau=8.0),
    MENU:         StagePolicy(floor=1, cap=6,  price=2.0, stake=1.0, consequence=1.0, tau=2.0),
    BALLOT:       StagePolicy(floor=1, cap=4,  price=2.0, stake=0.9, consequence=1.0, tau=2.0),
    WITNESS:      StagePolicy(floor=2, cap=8,  price=1.0, stake=1.2, consequence=2.0, tau=3.0),
    DELIBERATION: StagePolicy(floor=2, cap=6,  price=4.0, stake=1.0, consequence=1.5, tau=2.0),
}

#: Stakes tier -> multiplier applied to the VERIFICATION stages' stake only.
#: A flat multiplier would cancel out of every argmax and buy nothing; the
#: point of a stakes tier is to shift the SPLIT toward being caught being wrong.
TIER_MULTIPLIER: Mapping[str, float] = {
    "low": 0.6, "medium": 1.0, "high": 1.6, "critical": 2.4,
}


@dataclass(frozen=True)
class AllocatorPolicy:
    """The knobs. Deliberately small; every one is defended in a comment."""
    stages: Mapping[str, StagePolicy] = field(
        default_factory=lambda: dict(DEFAULT_POLICIES))
    tier_multiplier: Mapping[str, float] = field(
        default_factory=lambda: dict(TIER_MULTIPLIER))
    #: Fraction of the POST-FLOOR budget held uncommitted for preemption
    #: (design Q2: 15%). Released at the final checkpoint if unused.
    reserve_fraction: float = 0.15
    #: Design Q2: a challenger must beat the incumbent plan by 25%.
    hysteresis_ratio: float = 1.25
    #: ...on this many consecutive checkpoints. Two, so a single spike cannot
    #: move the plan and an oscillation can never accumulate a streak.
    consecutive_required: int = 2
    #: Uncertainty is clamped into [u_min, 1.0]. u_min > 0 keeps a stage's
    #: marginal score finite and ordered even at a perfect margin — a confident
    #: stage is deprioritized, never erased (see the module docstring).
    u_min: float = 0.05

    def __post_init__(self):
        if not (0.0 <= self.reserve_fraction < 1.0):
            raise AttentionError("reserve_fraction must be in [0, 1)")
        if self.hysteresis_ratio < 1.0:
            raise AttentionError("hysteresis_ratio must be >= 1.0")
        if self.consecutive_required < 1:
            raise AttentionError("consecutive_required must be >= 1")
        if not (0.0 < self.u_min <= 1.0):
            raise AttentionError("u_min must be in (0, 1]")
        unknown = set(self.stages) - set(STAGES)
        if unknown:
            raise AttentionError(f"unknown stage policies: {sorted(unknown)}")
        missing = set(STAGES) - set(self.stages)
        if missing:
            raise AttentionError(f"missing stage policies: {sorted(missing)}")


DEFAULT_POLICY = AllocatorPolicy()


# ----------------------------------------------------------------- task state
@dataclass(frozen=True)
class TaskState:
    """What the allocator is allowed to look at. Nothing else.

    `uncertainty` is a calibrated per-stage number in [0, 1] (1 = no idea).
    `margins` is the raw top-2 logprob gap per stage; a stage present in
    `margins` but absent from `uncertainty` is converted by
    `uncertainty_from_margin`. Absent from both -> `default_uncertainty`.

    `not_applicable` marks stages the seam manifest says do not exist for this
    task. It is the ONLY way a floor reaches zero, and it is a manifest
    statement about the task, never an inference from confidence.
    """
    budget: float
    stakes_tier: str = "medium"
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    margins: Mapping[str, float] = field(default_factory=dict)
    not_applicable: frozenset = frozenset()
    #: A hard-invariant violation observed elsewhere in the task. Bypasses
    #: hysteresis at the next checkpoint (design Q2) and pins the verification
    #: closure to the top of the priority order.
    safety_violation: bool = False
    #: Free-shape identity for the event record. Never read by the arithmetic.
    task_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            raise AttentionError(f"budget must be a number, got {self.budget!r}")
        if self.budget < 0 or not math.isfinite(self.budget):
            raise AttentionError(f"budget must be finite and >= 0, got {self.budget}")
        if self.stakes_tier not in STAKES_TIERS:
            raise AttentionError(
                f"unknown stakes_tier {self.stakes_tier!r}; "
                f"governed vocabulary is {list(STAKES_TIERS)}")
        for name in ("uncertainty", "margins"):
            m = getattr(self, name)
            if not isinstance(m, Mapping):
                raise AttentionError(f"{name} must be a mapping, got {m!r}")
            unknown = set(m) - set(STAGES)
            if unknown:
                raise AttentionError(f"{name} has unknown stages: {sorted(unknown)}")
            for k, v in m.items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise AttentionError(f"{name}[{k}] must be a number, got {v!r}")
                if not math.isfinite(v):
                    raise AttentionError(f"{name}[{k}] must be finite, got {v!r}")
        na = frozenset(self.not_applicable)
        unknown = na - set(STAGES)
        if unknown:
            raise AttentionError(f"not_applicable has unknown stages: {sorted(unknown)}")
        object.__setattr__(self, "not_applicable", na)


def uncertainty_from_margin(margin: float, tau: float = 0.15) -> float:
    """Top-2 logprob gap -> uncertainty in [0, 1].

    A margin at or above `tau` is "decisive" and maps to 0; a zero (or
    negative) margin maps to 1. Linear in between because we have no evidence
    for any other shape and a made-up curve would be a made-up claim.

    NOTE the standing warning: 0 here means *not fragile*, and never means
    *right*. The floors, not this number, are what protect verification.
    """
    if tau <= 0:
        raise AttentionError(f"tau must be positive, got {tau}")
    if margin <= 0:
        return 1.0
    if margin >= tau:
        return 0.0
    return 1.0 - (margin / tau)


def _effective_uncertainty(state: TaskState, stage: str,
                           policy: AllocatorPolicy,
                           default_uncertainty: float) -> float:
    if stage in state.uncertainty:
        u = float(state.uncertainty[stage])
    elif stage in state.margins:
        u = uncertainty_from_margin(float(state.margins[stage]))
    else:
        u = default_uncertainty
    return min(1.0, max(policy.u_min, u))


# ---------------------------------------------------------------- allocation
@dataclass(frozen=True)
class Allocation:
    """The plan. Immutable; a re-plan is a NEW Allocation, never an edit."""
    per_stage: Mapping[str, int]
    basis: Mapping[str, Any]

    #: "allocated" | "escalate". Escalate means the floors did not fit and the
    #: kernel must not execute a weakened plan.
    @property
    def verdict(self) -> str:
        return self.basis["verdict"]

    @property
    def spent(self) -> float:
        return self.basis["spent"]

    @property
    def reserve(self) -> float:
        return self.basis["reserve"]

    def to_dict(self) -> dict:
        return {"per_stage": dict(self.per_stage), "basis": dict(self.basis)}

    # ------------------------------------------------- DecisionEvent-ready
    def to_event_dict(self, correlation_id: str, *,
                      causation_id: str | None = None,
                      bundle_fingerprint: str | None = None) -> dict:
        """kwargs for `DecisionEvent(**d)` — the allocation as a stream record.

        `decision_type` reuses the governed vocabulary rather than inventing a
        word: an allocation IS an immutable plan (`plan_registered`), and a
        fail-closed floor shortfall IS an `escalate`. Nothing new is minted
        here; this module does not own that vocabulary.
        """
        if not correlation_id:
            raise AttentionError("correlation_id is required for an event")
        escalated = self.verdict == "escalate"
        options = tuple(
            {"option": s,
             "score": round(float(self.basis["scores"][s]), 6),
             "margin": round(float(self.basis["headroom"][s]), 6)}
            for s in STAGES
        )
        return {
            "correlation_id": correlation_id,
            "causation_id": causation_id,
            "producer": Producer(module="detreason.attention",
                                 version=MODULE_VERSION,
                                 bundle_fingerprint=bundle_fingerprint),
            "decision_type": ESCALATE if escalated else PLAN_REGISTERED,
            "inputs_refs": (content_ref(self.basis["state_ref_text"]),),
            "options_considered": options,
            "choice": dict(self.per_stage),
            "context": {"kind": "attention_allocation",
                        "key": self.basis["task_key"],
                        "basis": dict(self.basis)},
        }


def _state_ref_text(state: TaskState, policy: AllocatorPolicy) -> str:
    """Deterministic textual identity of the inputs, for content addressing."""
    parts = [f"budget={state.budget!r}", f"tier={state.stakes_tier}",
             f"safety={state.safety_violation}",
             f"na={sorted(state.not_applicable)}",
             f"reserve={policy.reserve_fraction!r}",
             f"hyst={policy.hysteresis_ratio!r}/{policy.consecutive_required}"]
    for s in STAGES:
        parts.append(f"{s}:u={state.uncertainty.get(s)!r},m={state.margins.get(s)!r}")
    return "|".join(parts)


# ------------------------------------------------------------------ the core
def _priority(state: TaskState, policy: AllocatorPolicy,
              default_uncertainty: float) -> tuple[dict, dict]:
    """A_s / P_s per stage, plus the effective uncertainties that produced it."""
    tier_mult = policy.tier_multiplier[state.stakes_tier]
    scores: dict[str, float] = {}
    used_u: dict[str, float] = {}
    for s in STAGES:
        sp = policy.stages[s]
        u = _effective_uncertainty(state, s, policy, default_uncertainty)
        used_u[s] = u
        if s in state.not_applicable:
            scores[s] = 0.0
            continue
        stake = sp.stake * (tier_mult if s in VERIFICATION_STAGES else 1.0)
        if state.safety_violation and s in VERIFICATION_STAGES:
            # A hard violation elsewhere pins the verification closure to the
            # top of the order. It does not change any floor and cannot take
            # another stage below its own.
            stake *= 4.0
        scores[s] = (stake * u * sp.consequence) / sp.price
    return scores, used_u


def _marginal(base: float, k: int, tau: float) -> float:
    return base / (1.0 + (k / tau))


def _waterfill(scores: Mapping[str, float], depths: dict[str, int],
               policy: AllocatorPolicy, spendable: float,
               applicable: tuple[str, ...]) -> tuple[dict[str, int], float]:
    """Hand out one atomic unit at a time to the highest marginal score.

    Deterministic: ties break on STAGE_INDEX. Terminates because every
    iteration either spends a strictly positive price or exits.
    """
    depths = dict(depths)
    left = spendable
    while True:
        best = None
        best_score = 0.0
        for s in applicable:
            sp = policy.stages[s]
            if depths[s] >= sp.cap or sp.price > left:
                continue
            m = _marginal(scores[s], depths[s] - sp.floor, sp.tau)
            if m <= 0.0:
                continue
            if best is None or m > best_score or (
                    m == best_score and STAGE_INDEX[s] < STAGE_INDEX[best]):
                best, best_score = s, m
        if best is None:
            return depths, left
        depths[best] += 1
        left -= policy.stages[best].price


def allocate(task_state: TaskState, policy: AllocatorPolicy = DEFAULT_POLICY,
             *, default_uncertainty: float = 0.5) -> Allocation:
    """Plan the budget. Pure, deterministic, offline.

    Order is the safety property: floors first and unconditionally, THEN the
    reserve is withheld, THEN whatever remains is water-filled by priority.
    """
    if not isinstance(task_state, TaskState):
        raise AttentionError("task_state must be a TaskState")
    if not isinstance(policy, AllocatorPolicy):
        raise AttentionError("policy must be an AllocatorPolicy")

    applicable = tuple(s for s in STAGES if s not in task_state.not_applicable)
    floors = {s: (0 if s in task_state.not_applicable else policy.stages[s].floor)
              for s in STAGES}
    floor_cost = sum(policy.stages[s].price * floors[s] for s in STAGES)

    scores, used_u = _priority(task_state, policy, default_uncertainty)
    state_text = _state_ref_text(task_state, policy)

    def _basis(verdict: str, depths: Mapping[str, int], spent: float,
               reserve: float, extra: Mapping[str, Any] | None = None) -> dict:
        b = {
            "verdict": verdict,
            "budget": float(task_state.budget),
            "spent": float(spent),
            "reserve": float(reserve),
            "floor_cost": float(floor_cost),
            "floors": dict(floors),
            "scores": dict(scores),
            "uncertainty": dict(used_u),
            "headroom": {s: float(policy.stages[s].cap - depths[s]) for s in STAGES},
            "stakes_tier": task_state.stakes_tier,
            "tier_multiplier": policy.tier_multiplier[task_state.stakes_tier],
            "safety_violation": bool(task_state.safety_violation),
            "not_applicable": sorted(task_state.not_applicable),
            "checkpoint": 0,
            "pending_challenge": None,
            "adopted": True,
            "reason": "initial_allocation",
            "task_key": task_state.task_key,
            "state_ref_text": state_text,
        }
        if extra:
            b.update(extra)
        return b

    # ---- fail CLOSED. The floors are not negotiable, so the budget is.
    if floor_cost > task_state.budget:
        depths = dict(floors)
        return Allocation(
            per_stage=depths,
            basis=_basis("escalate", depths, floor_cost, 0.0,
                         {"reason": "floors_exceed_budget",
                          "shortfall": float(floor_cost - task_state.budget)}),
        )

    post_floor = task_state.budget - floor_cost
    reserve = post_floor * policy.reserve_fraction
    spendable = post_floor - reserve

    depths, unspent = _waterfill(scores, floors, policy, spendable, applicable)
    spent = sum(policy.stages[s].price * depths[s] for s in STAGES)
    # Whatever water-filling could not place (nothing affordable, everything
    # capped) rejoins the reserve rather than evaporating into a false claim
    # that the budget was consumed.
    return Allocation(per_stage=depths,
                      basis=_basis("allocated", depths, spent,
                                   reserve + unspent))


# --------------------------------------------------------------- re-planning
def reallocate(previous: Allocation, task_state: TaskState,
               policy: AllocatorPolicy = DEFAULT_POLICY,
               *, default_uncertainty: float = 0.5) -> Allocation:
    """Checkpoint re-plan with hysteresis. Pure; `previous` is never mutated.

    A candidate plan is computed from the fresh state. It is ADOPTED only if:

      * the standing plan no longer fits the budget (it must move), or
      * a safety violation is declared (bypasses hysteresis, design Q2), or
      * the challenger's priority beats the standing plan's by
        `hysteresis_ratio` on `consecutive_required` consecutive checkpoints.

    Otherwise the standing plan is carried forward with the challenge streak
    recorded in `basis['pending_challenge']`. An oscillating signal resets the
    streak every other checkpoint and therefore never moves the allocation.
    """
    if not isinstance(previous, Allocation):
        raise AttentionError("previous must be an Allocation")
    candidate = allocate(task_state, policy,
                         default_uncertainty=default_uncertainty)
    checkpoint = int(previous.basis.get("checkpoint", 0)) + 1
    prev_scores = previous.basis["scores"]
    new_scores = candidate.basis["scores"]
    prev_pending = previous.basis.get("pending_challenge")

    def _finish(base: Allocation, adopted: bool, reason: str,
                pending: Any) -> Allocation:
        basis = dict(base.basis)
        basis.update({"checkpoint": checkpoint, "adopted": adopted,
                      "reason": reason, "pending_challenge": pending,
                      "previous_per_stage": dict(previous.per_stage),
                      # A carried-forward plan keeps the basis it was computed
                      # from, so record the budget observed AT THIS checkpoint
                      # separately rather than silently restating a stale one.
                      "checkpoint_budget": float(task_state.budget)})
        return replace(base, basis=basis)

    # The standing plan cannot be carried forward if it is unaffordable now,
    # or if it was itself an escalate, or if applicability changed.
    if previous.verdict == "escalate" or candidate.verdict == "escalate":
        return _finish(candidate, True, "escalate_path", None)
    if previous.spent > task_state.budget:
        return _finish(candidate, True, "previous_plan_unaffordable", None)
    if sorted(previous.per_stage) != sorted(candidate.per_stage) or any(
            previous.per_stage[s] < candidate.basis["floors"][s] for s in STAGES):
        # Floors are the one thing that may rise without hysteresis: a plan
        # below a floor is not a plan, it is a starved stage.
        return _finish(candidate, True, "floor_violation", None)
    if task_state.safety_violation:
        return _finish(candidate, True, "safety_violation_bypass", None)

    if dict(previous.per_stage) == dict(candidate.per_stage):
        return _finish(previous, False, "unchanged", None)

    # Challenger = the stage whose priority rose most against the standing
    # plan's basis. Ratio uses priorities, not depths, so the gate is about
    # evidence moving rather than about arithmetic rounding.
    challenger, ratio = None, 0.0
    for s in STAGES:
        old = prev_scores.get(s, 0.0)
        new = new_scores.get(s, 0.0)
        if new <= 0.0:
            continue
        r = math.inf if old <= 0.0 else new / old
        if challenger is None or r > ratio or (
                r == ratio and STAGE_INDEX[s] < STAGE_INDEX[challenger]):
            challenger, ratio = s, r

    if challenger is None or ratio < policy.hysteresis_ratio:
        return _finish(previous, False, "hysteresis_hold", None)

    streak = 1
    if isinstance(prev_pending, Mapping) and prev_pending.get("stage") == challenger:
        streak = int(prev_pending.get("streak", 0)) + 1
    if streak >= policy.consecutive_required:
        return _finish(candidate, True, "challenger_adopted",
                       {"stage": challenger, "streak": streak,
                        "ratio": None if ratio == math.inf else ratio})
    return _finish(previous, False, "challenger_pending",
                   {"stage": challenger, "streak": streak,
                    "ratio": None if ratio == math.inf else ratio})
