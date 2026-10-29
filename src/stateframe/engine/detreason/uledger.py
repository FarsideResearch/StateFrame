"""U-LEDGER — the uncertainty engine's shared bus (trunk faculty).

Design authority: sol_answers/a_deliberation.md Q5 (typed U-vector, accounting
discipline, closure only by evidence) and sol_answers/a_brain_census.md Q3 item 1
(U-ledger as every engine's bus: append-only typed events + firewalled working
memory).

The three load-bearing commitments, in code:

1. TYPED VECTOR, POLICY-WEIGHTED SCALAR. "Types are not naturally commensurable.
   Decision theory makes them comparable only after stakes and utility are
   applied." -> `ULedger.total(policy)` returns (vector, scalar, hard_max
   violations). The vector is for diagnosis, the scalar for scheduling/stopping,
   the hard maxima so a single critical row can veto a low scalar.

2. ACCOUNTING DISCIPLINE, NOT PHYSICS.
       U_next = U - verified_reduction + newly_exposed + transferred
   Every one of those four terms is a literal ledger operation (`close`,
   `expose`, `transfer`) writing an append-only event with its own typed delta.
   `equation_check()` replays the log and proves the identity to the micro-unit.

3. NO ROW CLOSES BECAUSE A MODEL ASSERTED CLOSURE. A `ClosureRecord` must name
   one of four criteria and carry the record that criterion demands. There is no
   `model_said_so` criterion, and a closure whose evidence does not satisfy its
   own criterion is REJECTED (`ClosureRejected`), not downgraded. This is the
   file's most important line of defence and its most important test.

Uncertainty is stored internally in integer micro-units so the accounting
identity is exact rather than float-approximate.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------------- vocabulary

UNKNOWN_ATOM = "unknown_atom"
THIN_MARGIN = "thin_margin"
UNGROUNDED_SYMBOL = "ungrounded_symbol"
UNVERIFIED_CLAIM = "unverified_claim"
UNEXPLORED_BRANCH = "unexplored_branch"
PREDICTED_RISK = "predicted_risk"

UTYPES = (UNKNOWN_ATOM, THIN_MARGIN, UNGROUNDED_SYMBOL, UNVERIFIED_CLAIM,
          UNEXPLORED_BRANCH, PREDICTED_RISK)

# The four ways a row may close. There is deliberately no model-assertion member.
DETERMINISTIC_CHECK = "deterministic_check"
INDEPENDENT_AUDIT = "independent_audit"
FALSIFICATION_PROBE = "falsification_probe"
EVIDENCE_SPAN = "evidence_span"

CLOSURE_CRITERIA = (DETERMINISTIC_CHECK, INDEPENDENT_AUDIT,
                    FALSIFICATION_PROBE, EVIDENCE_SPAN)

COST_CLASSES = {"free": 0.05, "cheap": 1.0, "moderate": 4.0,
                "expensive": 20.0, "human": 100.0}

_MICRO = 1_000_000


class ClosureRejected(Exception):
    """A close() attempt that did not satisfy a real closure criterion."""


class LedgerError(Exception):
    """Structural misuse of the ledger (unknown row, double close, ...)."""


def _micro(u: float) -> int:
    if u != u or u < 0.0 or u > 1.0:                 # NaN / out of range
        raise LedgerError(f"u must be a real number in [0,1], got {u!r}")
    return int(round(float(u) * _MICRO))


def _zero_vec() -> dict:
    return {t: 0 for t in UTYPES}


# ----------------------------------------------------------------------- pieces
@dataclass(frozen=True)
class ReductionAction:
    """One affordable move that would shrink this row. `expected_du` is a CLAIM;
    the ledger records what the action actually bought so the skeptic can compare."""
    action: str
    cost_class: str
    expected_du: float

    def __post_init__(self):
        if self.cost_class not in COST_CLASSES:
            raise LedgerError(f"unknown cost_class {self.cost_class!r}")
        if not (0.0 <= self.expected_du <= 1.0):
            raise LedgerError(f"expected_du out of range: {self.expected_du}")

    @property
    def voi(self) -> float:
        """Attention's VOI, wearing uncertainty units: expected du per unit cost."""
        return self.expected_du / COST_CLASSES[self.cost_class]


@dataclass
class ClosureRecord:
    """The evidence a closure stands on. Validated by `validate()` BEFORE any
    ledger state moves — a rejected closure leaves the row exactly as it was."""
    criterion: str
    actor: str = ""                  # who/what produced the closing evidence
    evidence_ids: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    residual_u: float = 0.0          # uncertainty that survives the closure
    ts: float = field(default_factory=time.time)

    def validate(self, row: "URow") -> None:
        c = self.criterion
        if c not in CLOSURE_CRITERIA:
            raise ClosureRejected(
                f"closure criterion {c!r} is not one of {CLOSURE_CRITERIA}; "
                "a model asserting closure is not a closure criterion")
        if row.closure_criterion and row.closure_criterion != c:
            raise ClosureRejected(
                f"row {row.id} declared closure criterion "
                f"{row.closure_criterion!r} but was closed with {c!r}")
        if not (0.0 <= self.residual_u <= row.u):
            raise ClosureRejected(
                f"residual_u {self.residual_u} must be within [0, {row.u}]")

        d = self.detail or {}
        if c == DETERMINISTIC_CHECK:
            # A named check that actually ran and actually passed.
            if not d.get("check_id"):
                raise ClosureRejected("deterministic_check needs check_id")
            if d.get("passed") is not True:
                raise ClosureRejected(
                    "deterministic_check closes a row only when passed is True")
        elif c == INDEPENDENT_AUDIT:
            # Independent means independent: the auditor may not be the owner.
            if not self.actor:
                raise ClosureRejected("independent_audit needs an auditor actor")
            if self.actor == row.owner:
                raise ClosureRejected(
                    f"independent_audit actor {self.actor!r} is the row owner; "
                    "self-audit is not independent")
            if d.get("verdict") != "confirmed":
                raise ClosureRejected(
                    "independent_audit closes only on verdict == 'confirmed'")
        elif c == FALSIFICATION_PROBE:
            # A probe that TRIED to break the claim and could not.
            if not d.get("probe_id"):
                raise ClosureRejected("falsification_probe needs probe_id")
            if "falsified" not in d:
                raise ClosureRejected(
                    "falsification_probe must report falsified True/False")
            if d.get("falsified") is not False:
                raise ClosureRejected(
                    "a probe that falsified the claim does not close the row")
        elif c == EVIDENCE_SPAN:
            if not self.evidence_ids:
                raise ClosureRejected("evidence_span needs evidence_ids")
            span = d.get("span")
            if not (isinstance(span, (list, tuple)) and len(span) == 2):
                raise ClosureRejected(
                    "evidence_span needs detail['span'] = [start, end]")
            if not d.get("source"):
                raise ClosureRejected("evidence_span needs detail['source']")

    def to_dict(self):
        return asdict(self)


@dataclass
class URow:
    """One enumerable piece of uncertainty attached to one decision."""
    id: str
    utype: str
    claim: str = ""
    severity: float = 0.5            # how much it matters if wrong [0,1]
    irreversibility: float = 0.0     # how permanent the damage would be [0,1]
    u: float = 1.0                   # normalized residual uncertainty [0,1]
    evidence_ids: list = field(default_factory=list)
    closure_criterion: str | None = None
    reduction_actions: list = field(default_factory=list)  # [ReductionAction]
    correlation_group: str = ""      # rows that fall together, fall together
    owner: str = ""
    source: str = ""                 # which engine put it on the bus
    opened_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    outcome: str | None = None       # closed | transferred | superseded
    closure: ClosureRecord | None = None

    def __post_init__(self):
        if self.utype not in UTYPES:
            raise LedgerError(f"unknown utype {self.utype!r}")
        if self.closure_criterion is not None and \
                self.closure_criterion not in CLOSURE_CRITERIA:
            raise ClosureRejected(
                f"declared closure_criterion {self.closure_criterion!r} invalid")
        for f in ("severity", "irreversibility", "u"):
            v = getattr(self, f)
            if not (v == v and 0.0 <= v <= 1.0):
                raise LedgerError(f"{f} out of range on row {self.id}: {v}")
        self.reduction_actions = [
            a if isinstance(a, ReductionAction) else ReductionAction(*a)
            for a in self.reduction_actions]

    @property
    def open(self) -> bool:
        return self.closed_at is None

    def best_action(self) -> ReductionAction | None:
        """Cheapest-biggest-du first: attention's VOI in uncertainty units."""
        return max(self.reduction_actions, key=lambda a: a.voi, default=None)

    def to_dict(self):
        d = asdict(self)
        d["reduction_actions"] = [asdict(a) for a in self.reduction_actions]
        d["closure"] = self.closure.to_dict() if self.closure else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "URow":
        d = dict(d)
        d["reduction_actions"] = [ReductionAction(**a)
                                  for a in d.get("reduction_actions", [])]
        cl = d.pop("closure", None)
        row = URow(**d)
        if cl:
            row.closure = ClosureRecord(**cl)
        return row


# ----------------------------------------------------------------------- policy
@dataclass(frozen=True)
class StakesPolicy:
    """Stakes turn an incommensurable vector into a comparable scalar, and set
    the hard maxima that no scalar may talk you out of."""
    name: str
    weights: dict
    threshold: float
    hard_max: dict = field(default_factory=dict)   # utype -> max open exposure

    def weight(self, utype: str) -> float:
        return float(self.weights.get(utype, 0.0))


_BASE_WEIGHTS = {
    UNKNOWN_ATOM: 0.30,
    UNGROUNDED_SYMBOL: 0.22,
    UNVERIFIED_CLAIM: 0.20,
    THIN_MARGIN: 0.14,
    UNEXPLORED_BRANCH: 0.08,
    PREDICTED_RISK: 0.06,
}

# a_deliberation Q5: "A reversible task may tolerate U~0.3; production or
# irreversible work targets U~0.02."
REVERSIBLE = StakesPolicy("reversible", dict(_BASE_WEIGHTS), 0.30,
                          {UNKNOWN_ATOM: 1.0})
STANDARD = StakesPolicy("standard", dict(_BASE_WEIGHTS), 0.10,
                        {UNKNOWN_ATOM: 0.50, UNVERIFIED_CLAIM: 0.60})
IRREVERSIBLE = StakesPolicy("irreversible", dict(_BASE_WEIGHTS), 0.02,
                            {UNKNOWN_ATOM: 0.0, UNGROUNDED_SYMBOL: 0.0,
                             UNVERIFIED_CLAIM: 0.10})

POLICIES = {p.name: p for p in (REVERSIBLE, STANDARD, IRREVERSIBLE)}


# ----------------------------------------------------------------------- ledger
class ULedger:
    """Append-only, per-decision-scoped uncertainty bus.

    Firewall (brain_census Q3.1): engines hold their own private working memory
    and BROADCAST here as typed events. Nothing mutates another engine's row —
    `transfer` moves a row to another decision's ledger by closing it here with
    an explicit transferred delta, so the audit trail never loses mass.
    """

    def __init__(self, decision_id: str, policy: StakesPolicy = STANDARD):
        self.decision_id = decision_id
        self.policy = policy
        self.rows: dict = {}
        self.events: list = []
        self._seq = 0

    # -- event log -----------------------------------------------------------
    def _emit(self, op: str, row_id: str, delta: dict, payload: dict) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, "decision_id": self.decision_id, "op": op,
              "row_id": row_id, "ts": time.time(),
              "delta_micro": {k: v for k, v in delta.items() if v},
              "payload": payload}
        self.events.append(ev)
        return ev

    # -- the four accounting operations --------------------------------------
    def open(self, row: URow) -> URow:
        """+ newly_exposed (at decision start this is the initial U)."""
        if row.id in self.rows:
            raise LedgerError(f"row {row.id} already on the ledger")
        self.rows[row.id] = row
        d = _zero_vec()
        d[row.utype] = _micro(row.u)
        self._emit("open", row.id, d,
                   {"utype": row.utype, "u": row.u, "claim": row.claim,
                    "source": row.source, "severity": row.severity,
                    "irreversibility": row.irreversibility,
                    "correlation_group": row.correlation_group})
        return row

    def expose(self, rows) -> list:
        """+ newly_exposed DISCOVERED DURING REDUCTION. Same write as open(),
        tagged so the skeptic can see that reducing U also revealed U — the term
        that keeps the equation honest instead of monotonically decreasing."""
        if isinstance(rows, URow):
            rows = [rows]
        out = []
        for row in rows:
            if row.id in self.rows:
                raise LedgerError(f"row {row.id} already on the ledger")
            self.rows[row.id] = row
            d = _zero_vec()
            d[row.utype] = _micro(row.u)
            self._emit("expose", row.id, d,
                       {"utype": row.utype, "u": row.u, "claim": row.claim,
                        "source": row.source,
                        "correlation_group": row.correlation_group})
            out.append(row)
        return out

    def close(self, row_id: str, closure: ClosureRecord) -> URow:
        """- verified_reduction. Raises ClosureRejected without touching state
        unless the closure record satisfies its own criterion.

        A closure with `residual_u > 0` is a PARTIAL verification: the original
        row closes for its full u (that is what the criterion actually bought),
        and the leftover is immediately re-exposed as a successor row. Leaving
        residual uncertainty attached to a closed row would be uncertainty
        nobody is carrying — the exact silent-drop this ledger exists to stop.
        """
        row = self.rows.get(row_id)
        if row is None:
            raise LedgerError(f"no such row {row_id}")
        if not row.open:
            raise LedgerError(f"row {row_id} already {row.outcome}")
        if not isinstance(closure, ClosureRecord):
            raise ClosureRejected(
                "close() requires a ClosureRecord carrying criterion evidence")
        closure.validate(row)                    # <- the load-bearing check

        reduced = _micro(row.u)
        residual = closure.residual_u
        row.closed_at = closure.ts
        row.outcome = "closed"
        row.closure = closure
        row.closure_criterion = closure.criterion
        row.evidence_ids = list(dict.fromkeys(
            list(row.evidence_ids) + list(closure.evidence_ids)))
        d = _zero_vec()
        d[row.utype] = -reduced
        self._emit("close", row_id, d,
                   {"criterion": closure.criterion, "actor": closure.actor,
                    "evidence_ids": closure.evidence_ids,
                    "residual_u": closure.residual_u,
                    "self_closed": bool(closure.actor)
                    and closure.actor == row.owner,
                    "detail": closure.detail})
        if residual > 0:
            succ = URow(
                id=f"{row_id}~residual",
                utype=row.utype,
                claim=f"residual after {closure.criterion} on {row_id}: {row.claim}",
                severity=row.severity, irreversibility=row.irreversibility,
                u=residual, evidence_ids=list(row.evidence_ids),
                closure_criterion=row.closure_criterion,
                reduction_actions=list(row.reduction_actions),
                correlation_group=row.correlation_group,
                owner=row.owner, source=f"{row.source}+residual")
            self.expose(succ)
        return row

    def transfer(self, row_id: str, to_decision: str,
                 to_ledger: "ULedger | None" = None) -> URow:
        """+/- transferred. The uncertainty does not evaporate because it moved:
        it leaves here with a negative delta and arrives there with a positive
        one, and both events name the same row id."""
        row = self.rows.get(row_id)
        if row is None:
            raise LedgerError(f"no such row {row_id}")
        if not row.open:
            raise LedgerError(f"row {row_id} already {row.outcome}")
        if to_decision == self.decision_id:
            raise LedgerError("cannot transfer a row to its own decision")
        moved = _micro(row.u)
        row.closed_at = time.time()
        row.outcome = "transferred"
        d = _zero_vec()
        d[row.utype] = -moved
        self._emit("transfer_out", row_id, d,
                   {"to_decision": to_decision, "u": row.u})
        if to_ledger is not None:
            if to_ledger.decision_id != to_decision:
                raise LedgerError("to_ledger.decision_id != to_decision")
            arrived = URow.from_dict(row.to_dict())
            arrived.closed_at, arrived.outcome, arrived.closure = None, None, None
            to_ledger.rows[arrived.id] = arrived
            din = _zero_vec()
            din[arrived.utype] = moved
            to_ledger._emit("transfer_in", arrived.id, din,
                            {"from_decision": self.decision_id, "u": arrived.u})
        return row

    # -- measurement ---------------------------------------------------------
    def vector(self) -> dict:
        """Typed U-vector over OPEN rows. For diagnosis, never for stopping."""
        v = {t: 0.0 for t in UTYPES}
        for r in self.rows.values():
            if r.open:
                v[r.utype] += r.u
        return {t: round(x, 6) for t, x in v.items()}

    def _exposure(self, row: URow, policy: StakesPolicy) -> float:
        """Risk exposure of one open row: uncertainty x how much being wrong
        costs x how permanent it is, in the policy's units."""
        irr = 1.0 + row.irreversibility
        return policy.weight(row.utype) * row.u * row.severity * irr

    def total(self, policy: StakesPolicy | None = None):
        """-> (typed vector, policy-weighted scalar, hard-max violations).

        The vector diagnoses, the scalar schedules, the violations veto.
        """
        policy = policy or self.policy
        vec = self.vector()
        scalar = 0.0
        per_type = {t: 0.0 for t in UTYPES}
        for r in self.rows.values():
            if not r.open:
                continue
            e = self._exposure(r, policy)
            scalar += e
            per_type[r.utype] += e
        violations = []
        for utype, cap in (policy.hard_max or {}).items():
            if per_type.get(utype, 0.0) > cap + 1e-12:
                violations.append({
                    "utype": utype, "exposure": round(per_type[utype], 6),
                    "hard_max": cap,
                    "rows": sorted(r.id for r in self.rows.values()
                                   if r.open and r.utype == utype)})
        return vec, round(scalar, 6), violations

    def ready_to_act(self, policy: StakesPolicy | None = None):
        """The stopping predicate. -> (bool, blocking_rows).

        Blocks on (a) any hard-max violation — one irreversible unknown vetoes
        even a tiny scalar — or (b) weighted exposure above the stakes threshold,
        in which case the biggest contributors are named as the blockers.
        """
        policy = policy or self.policy
        _, scalar, violations = self.total(policy)
        if violations:
            blocking = sorted({rid for v in violations for rid in v["rows"]})
            return False, blocking
        if scalar > policy.threshold + 1e-12:
            ranked = sorted((r for r in self.rows.values() if r.open),
                            key=lambda r: self._exposure(r, policy), reverse=True)
            blocking, acc = [], scalar
            for r in ranked:                      # the minimal set to get under
                if acc <= policy.threshold:
                    break
                blocking.append(r.id)
                acc -= self._exposure(r, policy)
            return False, blocking
        return True, []

    def next_action(self, policy: StakesPolicy | None = None):
        """Cheapest-biggest-dU next: -> (row_id, ReductionAction) or None."""
        policy = policy or self.policy
        best = None
        for r in self.rows.values():
            if not r.open:
                continue
            a = r.best_action()
            if a is None:
                continue
            score = a.voi * max(policy.weight(r.utype), 1e-9) * r.severity \
                * (1.0 + r.irreversibility)
            if best is None or score > best[0]:
                best = (score, r.id, a)
        return None if best is None else (best[1], best[2])

    # -- accounting proof ----------------------------------------------------
    def equation_check(self) -> dict:
        """Replay the log and prove U_next = U - reduced + exposed + transferred.

        Returns the four terms per type plus the residual against the live row
        state. `ok` is exact: everything is integer micro-units.
        """
        opened = _zero_vec()
        reduced = _zero_vec()
        exposed = _zero_vec()
        transferred = _zero_vec()
        running = _zero_vec()
        for ev in self.events:
            bucket = {"open": opened, "close": reduced, "expose": exposed,
                      "transfer_out": transferred,
                      "transfer_in": transferred}[ev["op"]]
            for t, dv in ev["delta_micro"].items():
                bucket[t] += abs(dv) if ev["op"] == "close" else dv
                running[t] += dv
        live = _zero_vec()
        for r in self.rows.values():
            if r.open:
                live[r.utype] += _micro(r.u)
        residual = {t: running[t] - live[t] for t in UTYPES}
        return {
            "opened": {t: opened[t] / _MICRO for t in UTYPES},
            "verified_reduction": {t: reduced[t] / _MICRO for t in UTYPES},
            "newly_exposed": {t: exposed[t] / _MICRO for t in UTYPES},
            "transferred": {t: transferred[t] / _MICRO for t in UTYPES},
            "computed": {t: running[t] / _MICRO for t in UTYPES},
            "live": {t: live[t] / _MICRO for t in UTYPES},
            "residual_micro": residual,
            "ok": all(v == 0 for v in residual.values()),
        }

    # -- Goodhart hooks for the skeptic --------------------------------------
    def close_rate_per_criterion(self) -> dict:
        """What the skeptic audits. Closure QUALITY, not closure count:
        a criterion that closes everything, closes it cheaply, and closes it
        with the row's own owner as actor is the Goodhart signature."""
        out = {c: {"closures": 0, "u_removed": 0.0, "self_closed": 0,
                   "residual_left": 0.0, "rows": [], "utypes": {}}
               for c in CLOSURE_CRITERIA}
        for ev in self.events:
            if ev["op"] != "close":
                continue
            c = ev["payload"]["criterion"]
            row = self.rows[ev["row_id"]]
            b = out[c]
            b["closures"] += 1
            b["u_removed"] += abs(sum(ev["delta_micro"].values())) / _MICRO
            b["self_closed"] += 1 if ev["payload"].get("self_closed") else 0
            b["residual_left"] += ev["payload"].get("residual_u", 0.0)
            b["rows"].append(ev["row_id"])
            b["utypes"][row.utype] = b["utypes"].get(row.utype, 0) + 1
        opened_total = sum(1 for e in self.events if e["op"] in ("open", "expose"))
        for c, b in out.items():
            b["u_removed"] = round(b["u_removed"], 6)
            b["residual_left"] = round(b["residual_left"], 6)
            b["share_of_opened"] = (round(b["closures"] / opened_total, 4)
                                    if opened_total else 0.0)
            b["mean_u_removed"] = (round(b["u_removed"] / b["closures"], 6)
                                   if b["closures"] else 0.0)
            b["self_closed_rate"] = (round(b["self_closed"] / b["closures"], 4)
                                     if b["closures"] else 0.0)
        return out

    def goodhart_flags(self) -> list:
        """Cheap structural tells the skeptic escalates on."""
        flags = []
        stats = self.close_rate_per_criterion()
        for c, b in stats.items():
            if b["closures"] and b["self_closed_rate"] > 0.0 \
                    and c != DETERMINISTIC_CHECK:
                flags.append({"flag": "self_closed_non_deterministic",
                              "criterion": c, "rate": b["self_closed_rate"]})
        # a correlation group closed wholesale by one criterion is one closure
        # wearing N hats, not N independent reductions.
        groups: dict = {}
        for r in self.rows.values():
            if r.correlation_group and not r.open and r.outcome == "closed":
                groups.setdefault((r.correlation_group, r.closure_criterion),
                                  []).append(r.id)
        for (grp, crit), ids in groups.items():
            if len(ids) >= 3:
                flags.append({"flag": "correlated_group_closed_together",
                              "correlation_group": grp, "criterion": crit,
                              "rows": sorted(ids)})
        return flags

    # -- persistence ---------------------------------------------------------
    def to_jsonl(self, path: str) -> str:
        with open(path, "w") as fh:
            fh.write(json.dumps({"_header": True,
                                 "decision_id": self.decision_id,
                                 "policy": self.policy.name}) + "\n")
            for r in self.rows.values():
                fh.write(json.dumps({"_row": r.to_dict()}) + "\n")
            for ev in self.events:
                fh.write(json.dumps({"_event": ev}) + "\n")
        return path

    @staticmethod
    def from_jsonl(path: str) -> "ULedger":
        led = None
        for line in open(path):
            rec = json.loads(line)
            if "_header" in rec:
                led = ULedger(rec["decision_id"],
                              POLICIES.get(rec["policy"], STANDARD))
            elif "_row" in rec:
                row = URow.from_dict(rec["_row"])
                led.rows[row.id] = row
            elif "_event" in rec:
                led.events.append(rec["_event"])
                led._seq = max(led._seq, rec["_event"]["seq"])
        if led is None:
            raise LedgerError(f"{path} has no ledger header")
        return led


# ------------------------------------------------------- ingestion adapters
# This is what makes the ledger a BUS and not an island: the two certificate
# producers that exist TODAY get read into U-rows without either of them
# knowing the ledger exists.

def _acts(*triples) -> list:
    return [ReductionAction(*t) for t in triples]


def from_fixer_cert(cert: dict, *, margin_tau: float = 0.15,
                    decision_id: str = "") -> list:
    """fixer.py certificate -> U-rows.

    * every margins row with margin < tau (or margin None, i.e. the probe never
      ran) -> thin_margin
    * ABSTAIN -> unknown_atom (the fixer told you it does not know)
    * LOW_CONFIDENCE_PROPOSAL / UNGATED_PROPOSAL -> unverified_claim
      (the evidence_linkage gate failed or never ran)
    * witness_replay with survivors > 1 -> unexplored_branch, one row per
      surviving-but-undiscriminated candidate set
    """
    key = cert.get("key") or cert.get("task") or "cert"
    fn = cert.get("fn") or ""
    base = decision_id or f"fixer:{key}:{fn}"
    rows: list = []

    for i, m in enumerate(cert.get("margins") or []):
        margin = m.get("margin")
        if margin is not None and margin >= margin_tau:
            continue
        # margin None == the top-2 logprob probe produced nothing; per kernel.py
        # that is zero CLAIMED confidence, so it is maximum thin-margin U.
        u = 1.0 if margin is None else max(0.0, (margin_tau - margin) / margin_tau)
        menu = m.get("menu", f"menu{i}")
        rows.append(URow(
            id=f"{base}#margin:{menu}:{i}",
            utype=THIN_MARGIN,
            claim=f"menu {menu!r} picked {m.get('pick')!r} out of "
                  f"{m.get('n')} with margin={margin}",
            severity=0.7 if menu in ("file", "function") else 0.5,
            irreversibility=0.0,
            u=round(u, 6),
            closure_criterion=DETERMINISTIC_CHECK,
            correlation_group=f"{base}/menu:{menu}",
            owner="fixer", source="fixer_cert",
            reduction_actions=_acts(
                ("reask_menu_reversed_order", "cheap", 0.6),
                ("anchor_by_exact_symbol", "cheap", 0.8),
                ("escalate_to_larger_model", "expensive", 0.9))))

    verdict = cert.get("verdict")
    if verdict == "ABSTAIN":
        reason = cert.get("reason") or _abstain_reason_from_stages(
            cert.get("stages") or [])
        rows.append(URow(
            id=f"{base}#abstain",
            utype=UNKNOWN_ATOM,
            claim=f"fixer abstained: {reason}",
            severity=0.9, irreversibility=0.0, u=1.0,
            closure_criterion=EVIDENCE_SPAN,
            correlation_group=f"{base}/abstain",
            owner="fixer", source="fixer_cert",
            reduction_actions=_acts(
                ("read_more_files", "cheap", 0.4),
                ("grep_definitions", "cheap", 0.5),
                ("runtime_probe", "moderate", 0.7),
                ("escalate_to_human", "human", 1.0))))
    elif verdict in ("LOW_CONFIDENCE_PROPOSAL", "UNGATED_PROPOSAL"):
        ungated = verdict == "UNGATED_PROPOSAL"
        rows.append(URow(
            id=f"{base}#unverified",
            utype=UNVERIFIED_CLAIM,
            claim=("evidence_linkage gate never ran on the proposed diff"
                   if ungated else
                   "proposed diff is not linked to the ticket's evidence"),
            severity=0.85, irreversibility=0.4,
            u=0.9 if ungated else 0.7,
            closure_criterion=DETERMINISTIC_CHECK,
            correlation_group=f"{base}/linkage",
            owner="fixer", source="fixer_cert",
            reduction_actions=_acts(
                ("run_evidence_linkage_gate", "cheap", 0.8),
                ("run_witness_replay_on_diff", "moderate", 0.9),
                ("human_review_diff", "human", 1.0))))

    for si, stage in enumerate(cert.get("stages") or []):
        wr = stage.get("witness_replay") if isinstance(stage, dict) else None
        if not wr:
            continue
        surv = wr.get("survivors") or 0
        if surv <= 1:
            continue
        rows.append(URow(
            id=f"{base}#branch:{si}",
            utype=UNEXPLORED_BRANCH,
            claim=f"{surv} of {wr.get('candidates')} mutation candidates "
                  f"survived witness replay ({wr.get('symptom_class')}); "
                  "no witness discriminates them",
            severity=0.6, irreversibility=0.2,
            u=round(1.0 - 1.0 / surv, 6),
            closure_criterion=FALSIFICATION_PROBE,
            correlation_group=f"{base}/witness_replay",
            owner="fixer", source="fixer_cert",
            reduction_actions=_acts(
                ("synthesize_discriminating_witness", "moderate", 0.8),
                ("run_existing_test_suite", "moderate", 0.5),
                ("deliberate_over_survivors", "expensive", 0.7))))
    return rows


def _abstain_reason_from_stages(stages: list) -> str:
    """Gym rows carry a trimmed certificate with no `reason`. The stage tail
    still says exactly how far the cascade got, so derive it rather than
    inventing one."""
    if not stages:
        return "no stages reached"
    last = stages[-1] if isinstance(stages[-1], dict) else {}
    if "mutations" in last and not last.get("mutations"):
        return "no mutation candidates generated for the symptom class"
    for k in ("function", "file", "class"):
        if k in last:
            return f"cascade stopped after {k}={last[k]!r}"
    return f"cascade stopped at stage {json.dumps(last)[:120]}"


def from_kernel_cert(certificate, *, decision_id: str = "") -> list:
    """detreason/kernel.py Certificate -> U-rows. UNKNOWN and CONFLICT atoms are
    exactly the kernel's own enumeration of what it could not resolve; CONFLICT
    is strictly worse than UNKNOWN (two sensors disagreed, so one is wrong)."""
    cert = certificate.to_dict() if hasattr(certificate, "to_dict") \
        else dict(certificate)
    base = decision_id or f"kernel:{cert.get('verdict', '?')}"
    pivotal = set(cert.get("pivotal_issues") or [])
    rows = []
    for aid, a in (cert.get("atoms") or {}).items():
        val = a.get("value")
        if val not in ("UNKNOWN", "CONFLICT"):
            continue
        conflict = val == "CONFLICT"
        is_pivotal = aid in pivotal
        rows.append(URow(
            id=f"{base}#atom:{aid}",
            utype=UNKNOWN_ATOM,
            claim=f"atom {aid} resolved {val} (sensor {a.get('sensor')!r}, "
                  f"margin {a.get('margin')})",
            severity=1.0 if is_pivotal else (0.8 if conflict else 0.6),
            irreversibility=0.0,
            u=1.0 if conflict else 0.85,
            closure_criterion=(INDEPENDENT_AUDIT if conflict else EVIDENCE_SPAN),
            correlation_group=f"{base}/atoms",
            owner="kernel", source="kernel_cert",
            reduction_actions=_acts(
                ("paired_entailment_probe", "cheap", 0.6),
                ("variant_verify", "cheap", 0.5),
                ("tool_sensor_if_expressible", "cheap", 0.95),
                ("escalate_to_human", "human", 1.0))))
    return rows


def ingest_certs(decision_id: str, certs, *, policy: StakesPolicy = STANDARD,
                 margin_tau: float = 0.15) -> ULedger:
    """Convenience: many fixer certificates -> one ledger (the nightly view).

    Certificates are enumerated because a run legitimately contains several
    certs for the SAME (key, fn) — different `kind`s, different configs. Keying
    row ids on (key, fn) alone silently collapsed 154 rows into 92 the first
    time this ran; the cert's index in the run is part of its identity.
    """
    led = ULedger(decision_id, policy)
    for i, c in enumerate(certs):
        base = (f"{decision_id}/{i}:{c.get('kind', 'cert')}:"
                f"{c.get('key', '?')}:{c.get('fn', '')}")
        for r in from_fixer_cert(c, margin_tau=margin_tau, decision_id=base):
            led.open(r)
    return led
