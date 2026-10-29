"""ADHERENCE — action binding, executable milestones, amendment protocol, focus.

PLANNING V1 MODULE 2. Design authority: ``sol_answers``-style banked specs at
``/tmp/solq/w14`` — ``a_plan_v1_cut.md`` (module 2), ``a_plan_amendment.md``
(anti-gaming protocol) and ``a_plan_focus.md`` (focus/tangent/thrash).

The problem this file exists to solve is NOT "did the agent do work" — the
ledger already answers that, and E38's self-coherence trap is exactly an agent
whose bookkeeping is immaculate while it wanders off the job. This file answers
four harder questions, all of them mechanically:

1. **Is this action even part of the plan?** Every proposed action binds to
   exactly one plan node, the current plan hash, and a mutation family, and its
   touched files/symbols must fall inside that node's frozen closure. The
   closure is computed from the node subtree's declared ``allowed_files`` /
   ``allowed_symbols`` UNIONED with the reverse-dependency ``impact()`` walk
   from ``detreason.relations`` — declared scope alone is either too tight
   (every legitimate ripple is a tangent) or hand-waved too wide.

2. **Is this milestone actually done?** A milestone is a tuple of EXECUTABLE
   acceptance predicates. Prose is rejected at plan registration, loudly, with
   :class:`ProseMilestoneRejected`. A milestone with no acceptance check is a
   milestone that can never fail, which is worse than no milestone.

3. **Is this re-plan legitimate, or is it drift-laundering?** The gaming attack
   (from the task-tracking crucible) is: amend the plan immediately before every
   unplanned action, so scope creep arrives as compliant paperwork. Three
   mechanisms defeat it and all three are required — amendments are legal only
   at milestone boundaries, budgets are hard and never reset (1/milestone,
   2/run), and divergence is ALWAYS measured against the immutable baseline
   ``P0``, never against the freshly-amended plan. Amendments can explain drift;
   they can never erase it.

4. **Is the agent stuck?** Same node, same canonical failure signature, same
   external-evidence digest, twice → the retry ladder escalates and ends in
   ABSTAIN. Never a loop. New external evidence is the ONLY thing that buys
   another honest attempt, and even that never rewinds the ladder.

Everything here is offline, deterministic, and pure: no clock, no filesystem,
no network, no LLM. Callers inject test runners and predicates. This module is
deliberately standalone — it does not import ``chains``, ``fixer`` or the work
ledger, so the plan-run wrapper can compose it without a cycle.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

# --------------------------------------------------------------------- verdicts

BOUND = "BOUND"
OUT_OF_SUBTREE = "OUT_OF_SUBTREE"
SPINE_VIOLATION = "SPINE_VIOLATION"
STALE_PLAN = "STALE_PLAN"
UNKNOWN_NODE = "UNKNOWN_NODE"
INACTIVE_NODE = "INACTIVE_NODE"
FORBIDDEN_FAMILY = "FORBIDDEN_FAMILY"

#: Every verdict a binding check can return. Anything not BOUND blocks the
#: action; the distinction exists so the focus policy can pick an interrupt
#: class and the divergence metric can weight the deviation.
BINDING_VERDICTS = (BOUND, OUT_OF_SUBTREE, SPINE_VIOLATION, STALE_PLAN,
                    UNKNOWN_NODE, INACTIVE_NODE, FORBIDDEN_FAMILY)

# --------------------------------------------------------------- acceptance kinds

TESTS_PASS = "tests_pass"
VIOLATIONS_ZERO = "violations_zero"
TREE_HASH = "tree_hash"
PREDICATE_NAME = "predicate_name"

#: The ONLY four acceptance kinds. There is deliberately no ``description`` or
#: ``prose`` member: a milestone whose acceptance cannot be executed is a
#: milestone that always passes.
ACCEPTANCE_KINDS = (TESTS_PASS, VIOLATIONS_ZERO, TREE_HASH, PREDICATE_NAME)

# ------------------------------------------------------------------ focus policy

HARD_STOP = "HARD_STOP"
FINISH_ATOM_THEN_REASSESS = "FINISH_ATOM_THEN_REASSESS"
STOP_BEFORE_ACTION = "STOP_BEFORE_ACTION"
PROCEED = "PROCEED"

#: Mutation families that may never ride out an in-flight atom. Discovering a
#: security/spine escape mid-action means stopping NOW, not "after this write".
HARD_STOP_FAMILIES = frozenset({"security", "spine", "credential",
                                "destructive", "production"})

# retry ladder rungs, in order. Monotonic: the level never decreases.
RETRY = "RETRY"
PROBE = "PROBE"
ESCALATE = "ESCALATE"
ABSTAIN = "ABSTAIN"
RETRY_LADDER = (RETRY, PROBE, ESCALATE, ABSTAIN)

#: Four atoms per node total, the probe included (a_plan_focus §3).
MAX_ATTEMPTS_PER_NODE = 4

# --------------------------------------------------------------------- amendments

LEGITIMATE = "LEGITIMATE"
SUSPECT_DRIFT = "SUSPECT_DRIFT"

#: Budgets. These never reset — not on resume, not on a child plan, not on a
#: new milestone beyond its own single allowance. Exhaustion means "ask a
#: human", not "start a fresh counter".
AMENDMENTS_PER_MILESTONE = 1
AMENDMENTS_PER_RUN = 2

#: Divergence thresholds (a_plan_amendment, deterministic metric).
DIVERGENCE_ALERT = 0.10
DIVERGENCE_FREEZE = 0.20
DIVERGENCE_WINDOW = 8


class AdherenceError(Exception):
    """Base for every hard refusal this module makes."""


class ProseMilestoneRejected(AdherenceError):
    """A milestone was registered without an executable acceptance predicate."""


class SpineAmendmentError(AdherenceError):
    """An amendment tried to move the immutable spine. Human gate required."""


class AmendmentRejected(AdherenceError):
    """An amendment was refused (budget, boundary, evidence, stale parent)."""


# ------------------------------------------------------------------- hashing

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def _sha(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()[:32]


# ========================================================================
# 1. ACTION BINDING
# ========================================================================

@dataclass(frozen=True)
class Action:
    """One proposed action, fully bound before it is allowed to run.

    ``files_touched``/``symbols_touched`` are the OBSERVED touch set supplied
    by the mechanical executor or tool adapter, never the model's own claim —
    a model that declares its touches is a model that can lie about them.
    An empty touch set for a mutating family is not "harmless", it is
    unobservable, and unobservable fails closed.
    """

    plan_node_id: str
    plan_hash: str
    files_touched: tuple = ()
    symbols_touched: tuple = ()
    mutation_family: str = "edit"
    action_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "files_touched",
                           tuple(sorted(self.files_touched)))
        object.__setattr__(self, "symbols_touched",
                           tuple(sorted(self.symbols_touched)))
        if not self.plan_node_id:
            raise ValueError("every action must name exactly one plan node")
        if not self.plan_hash:
            raise ValueError("every action must carry the plan hash it "
                             "believes it is executing under")

    @property
    def key(self) -> str:
        return self.action_id or _sha(
            (self.plan_node_id, self.plan_hash, self.files_touched,
             self.symbols_touched, self.mutation_family))


@dataclass(frozen=True)
class SpinePredicate:
    """One immutable-spine rule, expressed as an EXPLICIT predicate.

    The spine is the part of the plan no amendment may touch: workspace
    boundary, production/backup exclusions, security invariants, declared
    scope. Representing it as data (kind + args) rather than prose is what
    makes ``check_binding`` able to enforce it and ``propose_amendment``
    able to refuse to move it.

    Kinds:
      * ``forbid_files``      — literal paths that may never be touched.
      * ``forbid_prefixes``   — path prefixes (backups/, prod/) fenced off.
      * ``forbid_families``   — mutation families requiring a human gate.
      * ``confine_prefixes``  — EVERY touched file must sit under one of these.
    """

    name: str
    kind: str
    args: tuple = ()

    _KINDS = ("forbid_files", "forbid_prefixes", "forbid_families",
              "confine_prefixes")

    def __post_init__(self):
        if self.kind not in self._KINDS:
            raise ValueError(
                f"{self.kind!r} is not a spine predicate kind; "
                f"known: {self._KINDS}")
        object.__setattr__(self, "args", tuple(self.args))

    def violations(self, action: "Action") -> tuple:
        """Return the offending items, empty tuple when the action is clean."""
        if self.kind == "forbid_files":
            bad = set(self.args) & set(action.files_touched)
        elif self.kind == "forbid_prefixes":
            bad = {f for f in action.files_touched
                   if any(f.startswith(p) for p in self.args)}
        elif self.kind == "forbid_families":
            bad = ({action.mutation_family}
                   if action.mutation_family in self.args else set())
        else:  # confine_prefixes
            bad = {f for f in action.files_touched
                   if not any(f.startswith(p) for p in self.args)}
        return tuple(sorted(bad))


@dataclass(frozen=True)
class Acceptance:
    """One executable acceptance check: ``{kind, args}``. Prose need not apply."""

    kind: str
    args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ACCEPTANCE_KINDS:
            raise ProseMilestoneRejected(
                f"acceptance kind {self.kind!r} is not executable; "
                f"known kinds: {ACCEPTANCE_KINDS}")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "args": dict(self.args)}


@dataclass(frozen=True)
class Milestone:
    """A gate, not a label. At least one executable acceptance predicate."""

    id: str
    acceptance: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "acceptance", tuple(self.acceptance))
        if not self.acceptance:
            raise ProseMilestoneRejected(
                f"milestone {self.id!r} has no acceptance predicate — a "
                "milestone that cannot fail is not a gate")

    def as_dict(self) -> dict:
        return {"id": self.id,
                "acceptance": [a.as_dict() for a in self.acceptance]}


@dataclass(frozen=True)
class PlanNode:
    """One node of the plan tree, carrying its own declared scope."""

    id: str
    parent_id: str = ""
    allowed_files: tuple = ()
    allowed_symbols: tuple = ()
    mutation_families: tuple = ()
    milestone_id: str = ""

    def __post_init__(self):
        for f in ("allowed_files", "allowed_symbols", "mutation_families"):
            object.__setattr__(self, f, tuple(sorted(getattr(self, f))))

    def as_dict(self) -> dict:
        return {"id": self.id, "parent_id": self.parent_id,
                "allowed_files": list(self.allowed_files),
                "allowed_symbols": list(self.allowed_symbols),
                "mutation_families": list(self.mutation_families),
                "milestone_id": self.milestone_id}


@dataclass(frozen=True)
class Plan:
    """An immutable plan record. Amending produces a NEW Plan and a new hash."""

    plan_id: str
    nodes: tuple = ()
    milestones: tuple = ()
    spine: tuple = ()
    active_node_id: str = ""

    def __post_init__(self):
        object.__setattr__(self, "nodes",
                           tuple(sorted(self.nodes, key=lambda n: n.id)))
        object.__setattr__(self, "milestones",
                           tuple(sorted(self.milestones, key=lambda m: m.id)))
        object.__setattr__(self, "spine",
                           tuple(sorted(self.spine, key=lambda s: s.name)))

    # -- lookups ---------------------------------------------------------
    @property
    def node_index(self) -> dict:
        return {n.id: n for n in self.nodes}

    @property
    def milestone_index(self) -> dict:
        return {m.id: m for m in self.milestones}

    def children(self, node_id: str) -> tuple:
        return tuple(n for n in self.nodes if n.parent_id == node_id)

    def subtree(self, node_id: str) -> tuple:
        """Every node at or below ``node_id``, deterministic order.

        Cycle-safe: a malformed parent chain must not hang the executor.
        """
        idx = self.node_index
        if node_id not in idx:
            return ()
        out, seen, stack = [], {node_id}, [node_id]
        while stack:
            cur = stack.pop()
            out.append(idx[cur])
            for child in self.children(cur):
                if child.id not in seen:
                    seen.add(child.id)
                    stack.append(child.id)
        return tuple(sorted(out, key=lambda n: n.id))

    # -- identity --------------------------------------------------------
    @property
    def spine_hash(self) -> str:
        return _sha([(s.name, s.kind, list(s.args)) for s in self.spine])

    @property
    def plan_hash(self) -> str:
        return _sha({"plan_id": self.plan_id,
                     "nodes": [n.as_dict() for n in self.nodes],
                     "milestones": [m.as_dict() for m in self.milestones],
                     "spine_hash": self.spine_hash})


def register_plan(plan_id, nodes, milestones, spine=(), active_node_id="") -> Plan:
    """Build a Plan, REJECTING prose-only milestones at registration time.

    Rejection happens here rather than at gate time on purpose: a plan whose
    milestones cannot be executed must never reach the executor at all, or the
    agent spends hours doing work no gate will ever fail.
    """
    milestones = tuple(milestones)
    for m in milestones:
        if not isinstance(m, Milestone):
            raise ProseMilestoneRejected(
                f"milestone {m!r} is not a Milestone with executable "
                "acceptance predicates")
    nodes = tuple(nodes)
    known_milestones = {m.id for m in milestones}
    for n in nodes:
        if n.milestone_id and n.milestone_id not in known_milestones:
            raise AdherenceError(
                f"node {n.id!r} references unknown milestone "
                f"{n.milestone_id!r}")
    plan = Plan(plan_id=plan_id, nodes=nodes, milestones=milestones,
                spine=tuple(spine), active_node_id=active_node_id)
    if active_node_id and active_node_id not in plan.node_index:
        raise AdherenceError(f"active node {active_node_id!r} not in plan")
    return plan


# --------------------------------------------------------------- closure

@dataclass(frozen=True)
class Closure:
    """The frozen allowed-touch set for one node, plus its proof."""

    node_id: str
    files: tuple = ()
    symbols: tuple = ()
    derived_symbols: tuple = ()   # reached via relations.impact(), not declared

    @property
    def hash(self) -> str:
        return _sha({"node": self.node_id, "files": list(self.files),
                     "symbols": list(self.symbols)})


def _symbol_file(symbol_id: str) -> str:
    """``path::qualname`` -> ``path``. Matches ``relations.seam_node_id``."""
    return symbol_id.split("::", 1)[0] if "::" in symbol_id else ""


def compute_closure(plan: Plan, node_id: str, store=None, *, depth: int = 3,
                    expand_files_from_symbols: bool = True) -> Closure:
    """Allowed closure of ``node_id``: declared subtree scope + impact ripple.

    Declared scope alone is wrong in both directions. Too tight and every
    genuine consequence of an edit (the caller that must change with the
    signature) is scored a tangent, so the agent either stalls or the operator
    disables the check. Too loose and the check means nothing. So: seed the
    reverse-dependency walk with the subtree's declared symbols and take
    ``relations.impact()`` — the things that provably break if the seed
    changes — as in-scope.

    ``expand_files_from_symbols`` admits the DEFINING FILE of an impacted
    symbol. Without it the closure is self-defeating (a symbol is in scope but
    the only file you could edit it in is not). It is a flag, and it is
    explicit, because it is the one place this closure widens beyond what was
    declared. It never admits a file no impacted symbol lives in.
    """
    sub = plan.subtree(node_id)
    files = set()
    symbols = set()
    for n in sub:
        files.update(n.allowed_files)
        symbols.update(n.allowed_symbols)

    derived = set()
    if store is not None and symbols:
        for seed in sorted(symbols):
            for d in store.impact(seed, depth=depth):
                derived.add(d.target)
    # Impact targets are opaque ids: file-ish ones (no "::") widen the file
    # set, symbol-ish ones widen the symbol set.
    for target in derived:
        if "::" in target:
            symbols.add(target)
            if expand_files_from_symbols:
                owner = _symbol_file(target)
                if owner:
                    files.add(owner)
        else:
            files.add(target)

    return Closure(node_id=node_id,
                   files=tuple(sorted(files)),
                   symbols=tuple(sorted(symbols)),
                   derived_symbols=tuple(sorted(derived)))


# --------------------------------------------------------------- the check

@dataclass(frozen=True)
class BindingVerdict:
    """The answer, plus everything needed to explain and replay it."""

    verdict: str
    action_key: str = ""
    node_id: str = ""
    reasons: tuple = ()
    offending_files: tuple = ()
    offending_symbols: tuple = ()
    closure_hash: str = ""

    @property
    def bound(self) -> bool:
        return self.verdict == BOUND

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "action_key": self.action_key,
                "node_id": self.node_id, "reasons": list(self.reasons),
                "offending_files": list(self.offending_files),
                "offending_symbols": list(self.offending_symbols),
                "closure_hash": self.closure_hash}


def check_binding(action: Action, plan: Plan, store=None, *,
                  closure: Closure = None, depth: int = 3) -> BindingVerdict:
    """Is this action allowed to execute under this plan, right now?

    Order matters and is severity-first. The spine is checked BEFORE the plan
    hash: an action reaching for production or a credential is a spine
    violation whether or not its bookkeeping is current, and reporting it as
    a mere staleness problem would let a re-bind launder it.
    """
    reasons = []

    # 1. Spine. Unamendable, highest severity, checked first.
    spine_files, spine_syms = set(), set()
    for pred in plan.spine:
        bad = pred.violations(action)
        if bad:
            reasons.append(f"spine:{pred.name}")
            for item in bad:
                (spine_syms if item == action.mutation_family
                 else spine_files).add(item)
    if reasons:
        return BindingVerdict(SPINE_VIOLATION, action.key, action.plan_node_id,
                              tuple(reasons), tuple(sorted(spine_files)),
                              tuple(sorted(spine_syms)))

    # 2. Freshness. An action minted against a superseded plan must not run;
    #    that is precisely how a pre-amendment action gets laundered into the
    #    post-amendment plan.
    if action.plan_hash != plan.plan_hash:
        return BindingVerdict(STALE_PLAN, action.key, action.plan_node_id,
                              ("plan_hash_mismatch",))

    # 3. Node identity and activity.
    node = plan.node_index.get(action.plan_node_id)
    if node is None:
        return BindingVerdict(UNKNOWN_NODE, action.key, action.plan_node_id,
                              ("node_not_in_plan",))
    if plan.active_node_id and action.plan_node_id != plan.active_node_id:
        active_sub = {n.id for n in plan.subtree(plan.active_node_id)}
        if action.plan_node_id not in active_sub:
            return BindingVerdict(INACTIVE_NODE, action.key,
                                  action.plan_node_id,
                                  ("node_outside_active_subtree",))

    # 4. Closure. Computed from the ACTIVE node so a sibling's scope is never
    #    borrowed to justify a touch.
    anchor = plan.active_node_id or action.plan_node_id
    cl = closure if closure is not None else compute_closure(
        plan, anchor, store, depth=depth)

    bad_files = tuple(sorted(set(action.files_touched) - set(cl.files)))
    bad_syms = tuple(sorted(set(action.symbols_touched) - set(cl.symbols)))
    if bad_files or bad_syms:
        if bad_files:
            reasons.append("files_outside_closure")
        if bad_syms:
            reasons.append("symbols_outside_closure")
        return BindingVerdict(OUT_OF_SUBTREE, action.key, action.plan_node_id,
                              tuple(reasons), bad_files, bad_syms,
                              cl.hash)

    # 5. Mutation family.
    if node.mutation_families and \
            action.mutation_family not in node.mutation_families:
        return BindingVerdict(FORBIDDEN_FAMILY, action.key,
                              action.plan_node_id,
                              ("mutation_family_not_permitted",),
                              (), (action.mutation_family,), cl.hash)

    return BindingVerdict(BOUND, action.key, action.plan_node_id, (), (), (),
                          cl.hash)


# ========================================================================
# 2. MILESTONES — executable acceptance only
# ========================================================================

@dataclass(frozen=True)
class CheckResult:
    kind: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "passed": self.passed,
                "detail": self.detail}


@dataclass(frozen=True)
class MilestoneResult:
    milestone_id: str
    passed: bool
    checks: tuple = ()

    def as_dict(self) -> dict:
        return {"milestone_id": self.milestone_id, "passed": self.passed,
                "checks": [c.as_dict() for c in self.checks]}


def evaluate_milestone(milestone: Milestone, ctx: dict) -> MilestoneResult:
    """Run every acceptance predicate against the REAL post-state.

    ``ctx`` supplies the world:
      * ``test_runner(suite) -> bool``  for ``tests_pass``
      * ``violations(scope) -> list``   for ``violations_zero``
      * ``tree_hash() -> str``          for ``tree_hash``
      * ``predicates: {name: callable(args) -> bool}`` for ``predicate_name``

    A missing provider is a FAILED check, never a skipped one. "We could not
    check" and "it passed" must never be the same outcome — that equivalence is
    the whole self-coherence trap in miniature.
    """
    checks = []
    for acc in milestone.acceptance:
        checks.append(_run_acceptance(acc, ctx))
    return MilestoneResult(milestone.id,
                           all(c.passed for c in checks) and bool(checks),
                           tuple(checks))


def _run_acceptance(acc: Acceptance, ctx: dict) -> CheckResult:
    try:
        if acc.kind == TESTS_PASS:
            runner = ctx.get("test_runner")
            if runner is None:
                return CheckResult(acc.kind, False, "no test_runner provided")
            ok = bool(runner(acc.args.get("suite", "")))
            return CheckResult(acc.kind, ok,
                               f"suite={acc.args.get('suite','')}")

        if acc.kind == VIOLATIONS_ZERO:
            probe = ctx.get("violations")
            if probe is None:
                return CheckResult(acc.kind, False,
                                   "no violations probe provided")
            found = probe(acc.args.get("scope", ""))
            n = len(found or ())
            return CheckResult(acc.kind, n == 0, f"{n} violation(s)")

        if acc.kind == TREE_HASH:
            probe = ctx.get("tree_hash")
            if probe is None:
                return CheckResult(acc.kind, False, "no tree_hash provided")
            actual = probe()
            expected = acc.args.get("expected", "")
            return CheckResult(acc.kind, actual == expected,
                               f"actual={actual}")

        # predicate_name
        name = acc.args.get("name", "")
        fn = (ctx.get("predicates") or {}).get(name)
        if fn is None:
            return CheckResult(acc.kind, False, f"predicate {name!r} missing")
        return CheckResult(acc.kind, bool(fn(acc.args)), f"predicate={name}")
    except Exception as exc:                       # noqa: BLE001 — fail closed
        return CheckResult(acc.kind, False, f"raised: {type(exc).__name__}")


# ========================================================================
# 3. AMENDMENT PROTOCOL (anti-gaming)
# ========================================================================

@dataclass(frozen=True)
class Amendment:
    """A proposed plan change, carrying its own justification.

    ``evidence_refs`` must be EXTERNAL — a test receipt, a filesystem probe, an
    authenticated human instruction. The planner's own reasoning, the ledger,
    and the model's explanation do not qualify, because an agent that can mint
    its own evidence has no budget at all.
    """

    id: str
    reason_code: str
    old_plan_hash: str
    new_plan_hash: str
    evidence_refs: tuple = ()
    stakes: str = "S0"
    changed_node_ids: tuple = ()
    touches_spine: bool = False

    def __post_init__(self):
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "changed_node_ids",
                           tuple(sorted(self.changed_node_ids)))


@dataclass(frozen=True)
class AmendmentRecord:
    """Append-only audit row. Rejections are recorded, not swallowed."""

    amendment_id: str
    accepted: bool
    classification: str
    milestone_id: str
    reasons: tuple = ()
    budgets_after: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"amendment_id": self.amendment_id, "accepted": self.accepted,
                "classification": self.classification,
                "milestone_id": self.milestone_id,
                "reasons": list(self.reasons),
                "budgets_after": dict(self.budgets_after)}


# ========================================================================
# 4. THE RUN — binding + divergence + amendments + focus, one object
# ========================================================================

@dataclass
class AdherenceRun:
    """Live adherence state for one unattended run.

    Holds the IMMUTABLE baseline ``p0`` alongside the current (possibly
    amended) plan. Divergence is scored against ``p0`` forever. That single
    decision is what makes the gaming attack visible: amending the plan before
    each unplanned action changes ``plan`` but can never change ``p0``, so the
    divergence metric keeps climbing while the budget drains.
    """

    p0: Plan
    plan: Plan = None
    at_milestone_boundary: bool = False
    current_milestone_id: str = ""

    # append-only logs
    actions: list = field(default_factory=list)      # (verdict, delta)
    amendments: list = field(default_factory=list)   # AmendmentRecord
    events: list = field(default_factory=list)

    # counters — never reset
    _amendments_this_run: int = 0
    _amendments_by_milestone: dict = field(default_factory=dict)
    _thrash: dict = field(default_factory=dict)      # node_id -> _NodeState

    def __post_init__(self):
        if self.plan is None:
            self.plan = self.p0
        self._p0_nodes = {n.id for n in self.p0.nodes}
        self._p0_files = {f for n in self.p0.nodes for f in n.allowed_files}

    # -- events ----------------------------------------------------------
    def _emit(self, kind: str, **payload):
        self.events.append({"kind": kind, **payload})

    # ---------------------------------------------------------------- bind
    def admit(self, action: Action, store=None, *, closure: Closure = None,
              atomic_in_flight: bool = False) -> dict:
        """Check an action, score its divergence, and pick a focus decision.

        Returns a dict with the verdict, the interrupt decision, and the
        divergence state. Nothing here mutates the world; the caller decides
        whether to execute, and only when ``decision == PROCEED``.
        """
        verdict = check_binding(action, self.plan, store, closure=closure)
        delta = self._delta_for(action, verdict)
        self.actions.append({"action_key": action.key,
                             "verdict": verdict.verdict, "delta": delta,
                             "node_id": action.plan_node_id})
        decision = focus_decision(verdict, action,
                                  atomic_in_flight=atomic_in_flight)
        div = self.divergence()
        self._emit("ActionAdmitted" if verdict.bound else "TangentRejected",
                   action_key=action.key, verdict=verdict.verdict,
                   decision=decision, divergence=div)
        return {"verdict": verdict, "decision": decision, "delta": delta,
                "divergence": div, "divergence_state": self.divergence_state()}

    def _delta_for(self, action: Action, verdict: BindingVerdict) -> float:
        """Deterministic per-atom divergence, measured against ``P0``.

        0.0 — bound to a node that existed in the baseline.
        0.5 — bound, but to a node an amendment added (legal, still drift).
        1.0 — not bound at all: tangent, spine reach, stale, wrong family.
        """
        if not verdict.bound:
            return 1.0
        return 0.0 if action.plan_node_id in self._p0_nodes else 0.5

    def divergence(self) -> float:
        """``D_t = max(overall weighted mean, last-8 window mean)``.

        The window exists so a burst of drift late in a long clean run is not
        averaged into invisibility.
        """
        if not self.actions:
            return 0.0
        deltas = [a["delta"] for a in self.actions]
        overall = sum(deltas) / len(deltas)
        window = deltas[-DIVERGENCE_WINDOW:]
        recent = sum(window) / len(window)
        return max(overall, recent)

    def divergence_state(self) -> str:
        d = self.divergence()
        if d >= DIVERGENCE_FREEZE:
            return "FREEZE"
        if d >= DIVERGENCE_ALERT:
            return "ALERT"
        return "OK"

    # ----------------------------------------------------------- amendments
    def budgets(self) -> dict:
        mid = self.current_milestone_id
        return {
            "run_used": self._amendments_this_run,
            "run_limit": AMENDMENTS_PER_RUN,
            "milestone_used": self._amendments_by_milestone.get(mid, 0),
            "milestone_limit": AMENDMENTS_PER_MILESTONE,
        }

    def propose_amendment(self, amendment: Amendment,
                          new_plan: Plan = None) -> AmendmentRecord:
        """Accept or refuse a plan amendment. Refusals are recorded.

        Raises :class:`SpineAmendmentError` for a spine reach (that is not a
        budget question, it is a human gate) and :class:`AmendmentRejected`
        for every other refusal. Both leave an append-only record behind, so
        an exhausted budget cannot be hidden by catching the exception.
        """
        mid = self.current_milestone_id
        reasons = []

        # Spine is unamendable, full stop. Checked first, and it raises rather
        # than returning a verdict, because there is no autonomous recovery.
        if amendment.touches_spine or (
                new_plan is not None
                and new_plan.spine_hash != self.plan.spine_hash):
            rec = self._record(amendment, False, SUSPECT_DRIFT, mid,
                               ("spine_is_unamendable",))
            raise SpineAmendmentError(
                f"amendment {amendment.id!r} moves the immutable spine "
                f"(scope/workspace/backup-exclusion/security invariants). "
                f"This requires an authenticated HUMAN GATE; an agent may not "
                f"approve it. Audit row: {rec.as_dict()}")

        # Boundary. Amending mid-node is exactly the gaming attack's move.
        if not self.at_milestone_boundary:
            reasons.append("not_at_milestone_boundary")
        # External evidence, or it is drift wearing a form.
        if not amendment.evidence_refs:
            reasons.append("no_external_evidence")
        # Stale parent: an amendment must build on the plan actually running.
        if amendment.old_plan_hash != self.plan.plan_hash:
            reasons.append("stale_parent_plan_hash")
        # Budgets. Never reset, never refunded.
        if self._amendments_this_run >= AMENDMENTS_PER_RUN:
            reasons.append("run_budget_exhausted")
        if self._amendments_by_milestone.get(mid, 0) >= AMENDMENTS_PER_MILESTONE:
            reasons.append("milestone_budget_exhausted")

        if reasons:
            rec = self._record(amendment, False, SUSPECT_DRIFT, mid,
                               tuple(reasons))
            raise AmendmentRejected(
                f"amendment {amendment.id!r} refused: {', '.join(reasons)}")

        self._amendments_this_run += 1
        self._amendments_by_milestone[mid] = \
            self._amendments_by_milestone.get(mid, 0) + 1
        if new_plan is not None:
            self.plan = new_plan
        rec = self._record(amendment, True, LEGITIMATE, mid, ())
        self.at_milestone_boundary = False   # one amendment per boundary
        return rec

    def _record(self, amendment: Amendment, accepted: bool,
                classification: str, milestone_id: str,
                reasons: tuple) -> AmendmentRecord:
        rec = AmendmentRecord(amendment.id, accepted, classification,
                              milestone_id, reasons, self.budgets())
        self.amendments.append(rec)
        self._emit("AmendmentProposed", **rec.as_dict())
        return rec

    def suspect_amendment_count(self) -> int:
        return sum(1 for a in self.amendments
                   if a.classification == SUSPECT_DRIFT)

    # ---------------------------------------------------------------- focus
    def enter_milestone(self, milestone_id: str):
        """Open a milestone boundary — the only place amendments are legal."""
        self.current_milestone_id = milestone_id
        self.at_milestone_boundary = True
        self._emit("MilestoneBoundaryOpened", milestone_id=milestone_id)

    def record_failure(self, node_id: str, failure_signature: str,
                       evidence_hash: str) -> "ThrashVerdict":
        state = self._thrash.setdefault(node_id, _NodeState())
        verdict = state.record(failure_signature, evidence_hash)
        self._emit("RetryEscalated" if verdict.fired else "AttemptRecorded",
                   node_id=node_id, decision=verdict.decision,
                   level=verdict.level, attempts=verdict.attempts,
                   fired=verdict.fired)
        return verdict


# ------------------------------------------------------------ focus policy

def focus_decision(verdict: BindingVerdict, action: Action, *,
                   atomic_in_flight: bool = False) -> str:
    """Finish-atom-then-reassess by default; hard-stop on security/spine.

    The default is deliberate: aborting mid-write leaves the tree in a state
    neither the plan nor the resume protocol described. But the default has
    exceptions that are not negotiable — a spine violation or a
    security/destructive family never gets to finish what it started.
    """
    if verdict.verdict == SPINE_VIOLATION or \
            action.mutation_family in HARD_STOP_FAMILIES:
        return HARD_STOP
    if verdict.bound:
        return PROCEED
    return FINISH_ATOM_THEN_REASSESS if atomic_in_flight else STOP_BEFORE_ACTION


# ---------------------------------------------------------- thrash detector

@dataclass(frozen=True)
class ThrashVerdict:
    fired: bool
    decision: str
    level: int
    attempts: int
    reason: str = ""


@dataclass
class _NodeState:
    """Per-node retry ladder. Monotonic: the level never goes backwards.

    New external evidence buys another honest attempt but does NOT rewind the
    ladder — otherwise an agent that can produce a fresh-looking evidence
    digest each round has an infinite retry budget, which is the loop we are
    here to forbid.
    """

    attempts: int = 0
    level: int = 0
    last_signature: str = ""
    last_evidence: str = ""
    seen_evidence: set = field(default_factory=set)

    def record(self, failure_signature: str, evidence_hash: str) -> ThrashVerdict:
        self.attempts += 1
        same_failure = (self.attempts > 1
                        and failure_signature == self.last_signature)
        new_evidence = evidence_hash not in self.seen_evidence
        self.seen_evidence.add(evidence_hash)
        self.last_signature = failure_signature
        self.last_evidence = evidence_hash

        fired = bool(same_failure and not new_evidence)

        if self.attempts >= MAX_ATTEMPTS_PER_NODE:
            self.level = len(RETRY_LADDER) - 1        # ABSTAIN
        elif fired:
            self.level = min(self.level + 1, len(RETRY_LADDER) - 1)
        # a first attempt, or new evidence, leaves the level where it is

        decision = RETRY_LADDER[self.level]
        if self.attempts >= MAX_ATTEMPTS_PER_NODE:
            decision = ABSTAIN
        reason = ("same_failure_no_new_evidence" if fired else
                  "new_external_evidence" if new_evidence else
                  "first_attempt")
        return ThrashVerdict(fired, decision, self.level, self.attempts,
                             reason)


def failure_signature(node_id: str, acceptance: str, action_family: str,
                      normalized_failure: str) -> str:
    """Canonical failure signature ``F`` (a_plan_focus §3)."""
    return _sha((node_id, acceptance, action_family, normalized_failure))


def evidence_hash(*parts) -> str:
    """External-evidence digest ``E``. Only world-derived parts belong here."""
    return _sha(tuple(str(p) for p in parts))


__all__ = [
    "Action", "SpinePredicate", "Acceptance", "Milestone", "PlanNode", "Plan",
    "Closure", "BindingVerdict", "CheckResult", "MilestoneResult",
    "Amendment", "AmendmentRecord", "AdherenceRun", "ThrashVerdict",
    "register_plan", "compute_closure", "check_binding", "evaluate_milestone",
    "focus_decision", "failure_signature", "evidence_hash",
    "AdherenceError", "ProseMilestoneRejected", "SpineAmendmentError",
    "AmendmentRejected",
    "BOUND", "OUT_OF_SUBTREE", "SPINE_VIOLATION", "STALE_PLAN",
    "UNKNOWN_NODE", "INACTIVE_NODE", "FORBIDDEN_FAMILY", "BINDING_VERDICTS",
    "ACCEPTANCE_KINDS", "TESTS_PASS", "VIOLATIONS_ZERO", "TREE_HASH",
    "PREDICATE_NAME", "HARD_STOP", "FINISH_ATOM_THEN_REASSESS",
    "STOP_BEFORE_ACTION", "PROCEED", "HARD_STOP_FAMILIES",
    "RETRY", "PROBE", "ESCALATE", "ABSTAIN", "RETRY_LADDER",
    "LEGITIMATE", "SUSPECT_DRIFT", "AMENDMENTS_PER_MILESTONE",
    "AMENDMENTS_PER_RUN", "DIVERGENCE_ALERT", "DIVERGENCE_FREEZE",
    "DIVERGENCE_WINDOW", "MAX_ATTEMPTS_PER_NODE",
]
