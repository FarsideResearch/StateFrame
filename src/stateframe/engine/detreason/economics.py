"""KERNEL ECONOMICS v1 — decision-cost ledger, marginal-value rule, ESCALATE EV,
fleet throughput/cost reporting.

Design authority: sol_answers/w8/a_kernel_econ.md (Q1 decision-cost ledger +
marginal-value rule, Q2 ESCALATE expected-value formula, Q3 fleet scheduling).
This module implements Q1 and Q2 exactly as committed there, plus a Q3-adjacent
fleet *reporting* view (throughput_report) — not the GPU-slot scheduler itself,
which is out of scope for this file.

Ownership: this file + test_economics.py only. Everything else in detreason/
(uledger.py's StakesPolicy, results/unified_gym.jsonl) is read-only context.

------------------------------------------------------------------------------
DEVIATIONS FROM THE COMMITTED DESIGN (documented, not silent):

1. The design's VOI/marginal-value rule is stated over a risk-priced value-of-
   information functional R_s(x) = min_v E[L_s(v) | x], which requires a loss
   function and a belief state that this codebase's U-ledger (typed vector,
   not a full posterior over "v") does not yet expose in a form this module
   can consume without inventing one. We implement the OPERATIONAL form of the
   same rule that the design text also states directly and unconditionally:

       call the 1B only when  VOI(action) - C(action) > max(0, VOI(alt) - C(alt))

   with VOI supplied by the caller as `expected_info_gain` (a [0,1]-scaled
   estimate of uncertainty-reduction value, consistent with the U-ledger's
   [0,1] micro-unit uncertainty scale) and a stakes-scaled loss-avoided term
   folding in through `budget_state.loss_avoided`. This is a faithful
   inequality-level implementation of the committed rule; it is not the full
   Bayesian R_s(x) apparatus.

2. Budget cap: `B_s(d) = min(B_cap_s, rho_s * L_avoidable_d)` is implemented
   literally in `CostModel.budget_for_stakes`.

3. Competence priors for escalate_choice are INJECTED (p_success_retry,
   p_success_model, p_success_human), not derived from a live Beta posterior —
   the design's `theta_{a,k} ~ Beta(alpha,beta)` machinery lives in a
   different (not-yet-built) module. escalate_choice() computes EAC() over
   whatever priors it is given and is a pure function of them; when priors are
   missing it fails honest (stay_abstained) rather than guessing a prior.

4. `EAC(a) = C(a) + lambda_T*T(a) + pW(a)*L_FA_s + pA(a)*L_AB_s` is
   implemented literally, per candidate route. `remain abstained` is scored at
   EAC=0 cost / L_AB_s loss-of-never-resolving, per the design's arg-min set
   `{EAC(retry), EAC(model), EAC(human), EAC(remain_abstain)}`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Q1a. CostModel — per-action-class costs
# --------------------------------------------------------------------------

def calibrate_oneb_seconds(ledger_path: str) -> dict:
    """Recompute mean wall-seconds per 1B menu call from a gym-shaped JSONL
    ledger: sum(wall_s) / sum(len(margins)) across every row. Each `margins`
    entry in a gym row is one menu decision the harnessed 1B answered (file /
    function / mutation menus etc.); wall_s is the row's total wall time,
    dominated by 1B calls, so wall_s/margin-count is the right per-call
    estimator available from this kind of ledger (code checks are ~ms and
    don't materially skew the denominator).

    Pure function: reads `ledger_path` once, does no caching, writes nothing.
    Returns {"value": float, "rows": int, "margins": int, "ledger_path": str}.

    NOTE: `results/unified_gym.jsonl` is a LIVE file other processes append
    to (moving-corpus trap, repo rule #214 family) — calling this against
    that path will drift over time. `ONEB_CALL_CALIBRATION` below is a
    point-in-time SNAPSHOT of one such call, recorded with the row/margin
    counts it was taken at; it is not re-derived at import time and is not
    expected to match a fresh call of this function against the live file.
    """
    total_wall_s = 0.0
    total_margins = 0
    rows = 0
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            if row.get("wall_s") is not None:
                total_wall_s += row["wall_s"]
            total_margins += len(row.get("margins") or [])
    if total_margins == 0:
        raise ValueError(
            f"no margin probes found in {ledger_path!r}; cannot calibrate "
            "oneb_call seconds from an empty/malformed ledger")
    return {
        "value": total_wall_s / total_margins,
        "rows": rows,
        "margins": total_margins,
        "ledger_path": ledger_path,
    }


_GYM_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "unified_gym.jsonl",
)

# Point-in-time SNAPSHOT recorded 2026-08-28 via calibrate_oneb_seconds() over
# results/unified_gym.jsonl at 491 rows / 1136 margin probes:
#   33152.7 / 1136 == 29.18371478873239s
# This is a recorded estimate, NOT re-derived from the live file at import —
# the live file keeps growing (other processes append to it), so a fresh
# calibrate_oneb_seconds(_GYM_LEDGER_PATH) call will legitimately disagree
# with this snapshot as more rows land. Re-snapshot deliberately (update the
# dict below + this comment) when you want ONEB_CALL to track newer data;
# never assert this constant against a fresh read of the live ledger — see
# test_economics.py's calibration tests, which check the MATH against a
# frozen fixture and report live-vs-snapshot drift informationally only.
ONEB_CALL_CALIBRATION = {
    "value": 33152.7 / 1136,  # == 29.18371478873239
    "calibrated_at_rows": 491,
    "calibrated_at_margins": 1136,
    "ledger_path": _GYM_LEDGER_PATH,
    "calibrated_at": "2026-08-28",
}
_CALIBRATED_ONEB_CALL_SECONDS = ONEB_CALL_CALIBRATION["value"]


@dataclass(frozen=True)
class CostModel:
    """Per-action-class cost, in the ledger's four commensurable units:
    GPU-seconds (g), external dollars (m), human-minutes (h), latency-seconds (t).

    Matches sol_answers/w8/a_kernel_econ.md Q1's C(d) = sum_i(lambda_G g_i +
    lambda_$ m_i + lambda_H h_i + lambda_T t_i) decomposition — this dataclass
    IS the per-action (g, m, h, t) tuple for one action class; DecisionCostLedger
    applies the lambda weights and sums across stages.
    """
    gpu_seconds: float = 0.0
    dollars: float = 0.0
    human_minutes: float = 0.0
    latency_seconds: float = 0.0

    def weighted(self, lambda_g: float, lambda_m: float, lambda_h: float,
                 lambda_t: float) -> float:
        return (lambda_g * self.gpu_seconds + lambda_m * self.dollars +
                lambda_h * self.human_minutes + lambda_t * self.latency_seconds)


# code_check: ~0 cost (ms-scale, deterministic, no GPU/human/dollar spend).
CODE_CHECK = CostModel(gpu_seconds=0.0, dollars=0.0, human_minutes=0.0,
                        latency_seconds=0.001)

# oneb_call: calibrated from the real ledger above. GPU-seconds == latency
# here (the 1B call blocks on its own GPU-second cost; no separate queue wait
# folded in — that is Q3's fleet-scheduling concern, out of scope for this
# per-decision cost model).
ONEB_CALL = CostModel(gpu_seconds=_CALIBRATED_ONEB_CALL_SECONDS, dollars=0.0,
                       human_minutes=0.0,
                       latency_seconds=_CALIBRATED_ONEB_CALL_SECONDS)

# model_escalation: design commits "~$0.5-2, ~seconds-to-tens-of-seconds
# latency" as an order-of-magnitude, not a measured number (no model spend is
# recorded in results/unified_gym.jsonl — that ledger is 1B-only). Configurable.
MODEL_ESCALATION = CostModel(gpu_seconds=0.0, dollars=1.0, human_minutes=0.0,
                             latency_seconds=15.0)

# human_question: large constant expressing preciousness. No dollar cost
# modeled (humans aren't metered like an API) — the entire cost is
# human-minutes, weighted heavily by lambda_H below.
HUMAN_QUESTION = CostModel(gpu_seconds=0.0, dollars=0.0, human_minutes=5.0,
                            latency_seconds=600.0)

ACTION_CLASS_COSTS: dict[str, CostModel] = {
    "code_check": CODE_CHECK,
    "oneb_call": ONEB_CALL,
    "model_escalation": MODEL_ESCALATION,
    "human_question": HUMAN_QUESTION,
}

# lambda weights turning (g, m, h, t) into one scalar $-equivalent cost.
# lambda_H is deliberately large: "human ask = precious" (design Q1 framing).
DEFAULT_LAMBDA_GPU = 0.02       # $/GPU-second (rough on-demand GPU pricing)
DEFAULT_LAMBDA_DOLLAR = 1.0     # dollars are already dollars
DEFAULT_LAMBDA_HUMAN = 5.0      # $/human-minute — precious, not free
DEFAULT_LAMBDA_LATENCY = 0.001  # $/latency-second — small but nonzero (t_i)


def action_cost(action_class: str, n: int = 1,
                 lambda_g: float = DEFAULT_LAMBDA_GPU,
                 lambda_m: float = DEFAULT_LAMBDA_DOLLAR,
                 lambda_h: float = DEFAULT_LAMBDA_HUMAN,
                 lambda_t: float = DEFAULT_LAMBDA_LATENCY) -> float:
    """Scalar $-equivalent cost of `n` invocations of `action_class`."""
    if action_class not in ACTION_CLASS_COSTS:
        raise ValueError(f"unknown action_class: {action_class!r}")
    if n < 0:
        raise ValueError("n must be >= 0")
    unit = ACTION_CLASS_COSTS[action_class]
    return n * unit.weighted(lambda_g, lambda_m, lambda_h, lambda_t)


# --------------------------------------------------------------------------
# Q1b. DecisionCostLedger — append-only results/econ_ledger.jsonl
# --------------------------------------------------------------------------

_DEFAULT_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "econ_ledger.jsonl",
)


@dataclass
class DecisionCostLedger:
    """Append-only per-task charge log. Every `charge()` call writes one JSON
    line to `path` (default results/econ_ledger.jsonl) and updates the
    in-memory running totals for the task. Mirrors uledger.py's append-only
    discipline: nothing is charged silently, nothing is un-charged.
    """
    path: str = _DEFAULT_LEDGER_PATH
    lambda_g: float = DEFAULT_LAMBDA_GPU
    lambda_m: float = DEFAULT_LAMBDA_DOLLAR
    lambda_h: float = DEFAULT_LAMBDA_HUMAN
    lambda_t: float = DEFAULT_LAMBDA_LATENCY
    _totals: dict = field(default_factory=dict)   # task_id -> scalar $ total
    _events: list = field(default_factory=list)    # in-memory mirror, this process

    def charge(self, task_id: str, stage: str, action_class: str,
               n: int = 1, meta: Optional[dict] = None) -> float:
        """Charge `n` invocations of `action_class` at `stage` against
        `task_id`. Returns the scalar $-equivalent cost of this charge and
        appends both an in-memory event and a JSONL line to `self.path`.
        """
        if not task_id:
            raise ValueError("task_id is required")
        cost = action_cost(action_class, n, self.lambda_g, self.lambda_m,
                            self.lambda_h, self.lambda_t)
        event = {
            "task_id": task_id,
            "stage": stage,
            "action_class": action_class,
            "n": n,
            "cost": cost,
            "ts": time.time(),
            "meta": meta or {},
        }
        self._events.append(event)
        self._totals[task_id] = self._totals.get(task_id, 0.0) + cost
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(event) + "\n")
        return cost

    def task_total(self, task_id: str) -> float:
        return self._totals.get(task_id, 0.0)

    def task_events(self, task_id: str) -> list:
        return [e for e in self._events if e["task_id"] == task_id]

    @staticmethod
    def totals_from_file(path: str) -> dict:
        """Replay a ledger file and return {task_id: scalar_total}, without
        needing a live DecisionCostLedger instance. Used by throughput_report
        to read ledgers this process did not write.
        """
        totals: dict = {}
        if not os.path.exists(path):
            return totals
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                tid = event["task_id"]
                totals[tid] = totals.get(tid, 0.0) + event["cost"]
        return totals


# --------------------------------------------------------------------------
# Q1c. Marginal-value rule
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetState:
    """Per-task budget state, per design Q1:
        B_s(d) = min(B_cap_s, rho_s * L_avoidable_d)
    `spent` is what's already been charged against this task (from the
    ledger); `loss_avoidable` and `rho_s` (stakes multiplier) feed the budget
    cap. `hard_floor_blocked=True` means a kernel safety floor vetoes the call
    regardless of VOI (design: "budget plus kernel safety floors permit it").
    """
    spent: float = 0.0
    loss_avoidable: float = 0.0
    rho_s: float = 1.0
    budget_cap: float = float("inf")
    hard_floor_blocked: bool = False

    def budget(self) -> float:
        return min(self.budget_cap, self.rho_s * self.loss_avoidable)

    def remaining(self) -> float:
        return self.budget() - self.spent


def worth_calling(expected_info_gain: float, action_class: str,
                   budget_state: BudgetState, *,
                   alt_expected_info_gain: float = 0.0,
                   alt_action_class: str = "code_check",
                   n: int = 1,
                   lambda_g: float = DEFAULT_LAMBDA_GPU,
                   lambda_m: float = DEFAULT_LAMBDA_DOLLAR,
                   lambda_h: float = DEFAULT_LAMBDA_HUMAN,
                   lambda_t: float = DEFAULT_LAMBDA_LATENCY) -> bool:
    """Design Q1: call the 1B (or any `action_class`) only when

        VOI_s(action) - C(action) > max(0, VOI_s(alt) - C(alt))

    and budget + safety floors permit it. `expected_info_gain` and
    `alt_expected_info_gain` are VOI proxies on the U-ledger's [0,1] scale,
    scaled to the same $-equivalent units as cost via `budget_state.rho_s`
    (the stakes multiplier that also sets the budget cap) so the inequality
    is dimensionally consistent: VOI_dollars = expected_info_gain * rho_s.

    Deterministic checks settle first "whenever they can settle the atom or
    materially reduce uncertainty" — callers achieve this by trying
    `action_class="code_check"` first in their own control flow; this
    function only adjudicates ONE proposed call against ONE alternative.
    """
    if budget_state.hard_floor_blocked:
        return False

    cost = action_cost(action_class, n, lambda_g, lambda_m, lambda_h, lambda_t)
    alt_cost = action_cost(alt_action_class, n, lambda_g, lambda_m, lambda_h,
                            lambda_t)

    voi = expected_info_gain * budget_state.rho_s
    alt_voi = alt_expected_info_gain * budget_state.rho_s

    net = voi - cost
    alt_net = max(0.0, alt_voi - alt_cost)

    if net <= alt_net:
        return False

    # Budget gate: the call itself must fit in what's left of B_s(d).
    if cost > budget_state.remaining():
        return False

    return True


# --------------------------------------------------------------------------
# Q2. ESCALATE expected-value formula
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompetencePriors:
    """Injected p_success_* priors for one escalation route. The design's
    theta_{a,k} ~ Beta posterior machinery is a separate (not-yet-built)
    module; this dataclass is the interface point where those posteriors
    would eventually be summarized into point estimates. Until then, callers
    supply their own estimates (e.g. from historical gym pass rates).

    p_success = pC(a): correct result accepted after witness/seam replay.
    p_wrong   = pW(a): wrong result accepted.
    p_abstain = pA(a): remains unresolved / abstains.
    They need not sum to exactly 1.0 (float slop tolerated) but must be in
    [0,1] and should be close to summing to 1 — this is asserted loosely.
    """
    p_success: float
    p_wrong: float
    p_abstain: float

    def __post_init__(self):
        for name, v in (("p_success", self.p_success),
                         ("p_wrong", self.p_wrong),
                         ("p_abstain", self.p_abstain)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v!r}")


@dataclass(frozen=True)
class EscalationLosses:
    """Stakes-scaled losses used by EAC(). L_FA_s = loss of a wrong result
    being accepted at stakes tier s; L_AB_s = loss of the task remaining
    permanently abstained/unresolved at stakes tier s. Both should be
    calibrated per stakes tier by the caller (irreversible >> standard >>
    reversible), mirroring uledger.py's StakesPolicy escalation by tier.
    """
    l_false_accept: float
    l_abstain: float


ROUTE_RETRY = "retry_ladder"
ROUTE_MODEL = "model"
ROUTE_HUMAN = "human"
ROUTE_STAY_ABSTAINED = "stay_abstained"

_ROUTE_ACTION_CLASS = {
    ROUTE_RETRY: "oneb_call",
    ROUTE_MODEL: "model_escalation",
    ROUTE_HUMAN: "human_question",
}


def _eac(action_class: Optional[str], priors: Optional[CompetencePriors],
         losses: EscalationLosses, *, n: int = 1,
         lambda_g: float, lambda_m: float, lambda_h: float, lambda_t: float
         ) -> Optional[float]:
    """EAC(a) = C(a) + lambda_T*T(a) + pW(a)*L_FA_s + pA(a)*L_AB_s

    C(a) already folds in the action's latency via `action_cost`'s
    lambda_t*t term, so the design's separate `+ lambda_T*T(a)` term is
    represented once inside C(a) rather than double-counted; T(a) is the
    latency component of the SAME action, not an independent quantity. This
    keeps the formula's units consistent (one $-equivalent scalar) without
    double-billing latency.

    `stay_abstained` (action_class=None) has zero call cost, zero pW (nothing
    was accepted, so nothing can be wrongly accepted), pA=1.0 (it stays
    abstained by definition) -> EAC = L_AB_s exactly, matching the design's
    "if none beats remaining abstention, stay ABSTAIN" default.
    """
    if action_class is None:
        return losses.l_abstain
    if priors is None:
        return None
    cost = action_cost(action_class, n, lambda_g, lambda_m, lambda_h, lambda_t)
    return (cost + priors.p_wrong * losses.l_false_accept +
            priors.p_abstain * losses.l_abstain)


def escalate_choice(losses: EscalationLosses, *,
                     p_success_retry: Optional[CompetencePriors] = None,
                     p_success_model: Optional[CompetencePriors] = None,
                     p_success_human: Optional[CompetencePriors] = None,
                     n: int = 1,
                     lambda_g: float = DEFAULT_LAMBDA_GPU,
                     lambda_m: float = DEFAULT_LAMBDA_DOLLAR,
                     lambda_h: float = DEFAULT_LAMBDA_HUMAN,
                     lambda_t: float = DEFAULT_LAMBDA_LATENCY) -> dict:
    """Given an abstained task, choose among retry_ladder / model / human /
    stay_abstained by minimum expected-accept-cost (EAC), per design Q2:

        a* = argmin{EAC(retry), EAC(model), EAC(human), EAC(remain_abstain)}
        if none beats remaining abstention: stay ABSTAIN

    Honest-default rule: a route whose competence prior was not supplied
    (None) gets NO EAC entry at all — it is not silently assumed to be free
    or perfect. If EVERY escalation route is missing its prior, the only
    entry in ev_table is `stay_abstained`, and the choice is `stay_abstained`
    with an explicit `reason` explaining why (missing priors), not a guess.
    """
    ev_table: dict[str, Optional[float]] = {
        ROUTE_STAY_ABSTAINED: _eac(None, None, losses, n=n, lambda_g=lambda_g,
                                    lambda_m=lambda_m, lambda_h=lambda_h,
                                    lambda_t=lambda_t),
    }

    route_priors = {
        ROUTE_RETRY: p_success_retry,
        ROUTE_MODEL: p_success_model,
        ROUTE_HUMAN: p_success_human,
    }

    any_prior_supplied = False
    for route, priors in route_priors.items():
        if priors is None:
            continue
        any_prior_supplied = True
        ev_table[route] = _eac(_ROUTE_ACTION_CLASS[route], priors, losses,
                                n=n, lambda_g=lambda_g, lambda_m=lambda_m,
                                lambda_h=lambda_h, lambda_t=lambda_t)

    if not any_prior_supplied:
        return {
            "choice": ROUTE_STAY_ABSTAINED,
            "ev_table": ev_table,
            "reason": ("no competence priors supplied for any escalation "
                       "route; honest default is stay_abstained, not a "
                       "guessed EV comparison"),
        }

    # argmin over whatever routes have a real EAC entry (stay_abstained
    # always has one — see _eac(None, ...) above).
    choice = min(ev_table, key=lambda r: ev_table[r])
    return {"choice": choice, "ev_table": ev_table, "reason": None}


# --------------------------------------------------------------------------
# Q3-adjacent. Fleet throughput/cost report from real ledger data
# --------------------------------------------------------------------------

def throughput_report(ledger_paths: list, gym_path: Optional[str] = None
                       ) -> dict:
    """Fleet view: verified-fixes-per-hour and cost-per-verified-fix, computed
    from REAL data.

    Cost side: replays every path in `ledger_paths` (results/econ_ledger.jsonl
    by convention) via `DecisionCostLedger.totals_from_file` and sums the
    scalar $-equivalent totals across all tasks in all ledgers.

    Throughput side: since results/econ_ledger.jsonl does not yet exist in
    this repo (no kernel run has been wired to charge() yet — this module is
    v1), throughput is computed from `results/unified_gym.jsonl` (the real,
    measured gym ledger this task was told to use) when `gym_path` is given
    (default: results/unified_gym.jsonl next to this file's repo root).
    A "verified fix" in that ledger is `fixed == 1` (the gym's own
    ground-truth flag for a mutation whose proposed fix matched/repaired the
    injected defect and was accepted — see FIX_PROPOSED+fixed=1 and
    LOW_CONFIDENCE_PROPOSAL+fixed=1 rows). wall_s is real measured wall-clock
    seconds per task.
    """
    total_cost = 0.0
    ledgers_read = []
    for path in ledger_paths:
        totals = DecisionCostLedger.totals_from_file(path)
        total_cost += sum(totals.values())
        ledgers_read.append({"path": path, "task_count": len(totals),
                              "cost": sum(totals.values())})

    if gym_path is None:
        gym_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "unified_gym.jsonl")

    verified_fixes = 0
    total_wall_s = 0.0
    total_tasks = 0
    if os.path.exists(gym_path):
        with open(gym_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total_tasks += 1
                wall_s = row.get("wall_s")
                if wall_s is not None:
                    total_wall_s += wall_s
                if row.get("fixed") == 1:
                    verified_fixes += 1

    total_hours = total_wall_s / 3600.0
    fixes_per_hour = (verified_fixes / total_hours) if total_hours > 0 else 0.0
    cost_per_fix = (total_cost / verified_fixes) if verified_fixes > 0 else None

    return {
        "gym_path": gym_path,
        "gym_task_count": total_tasks,
        "verified_fixes": verified_fixes,
        "total_wall_s": total_wall_s,
        "total_wall_hours": total_hours,
        "verified_fixes_per_hour": fixes_per_hour,
        "total_cost": total_cost,
        "cost_per_verified_fix": cost_per_fix,
        "ledgers": ledgers_read,
    }
