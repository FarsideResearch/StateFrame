"""DETREASON SKEPTIC (v2) — the adversarial trust-nothing engine.

Design authority: sol_answers/a_skeptic.md (Q1 suspicion allocator, Q2 attack
move library, Q3 governance), detreason/kernel.py (Certificate shapes).

Two halves, mirroring the kernel's sense/judge split:

  1. SUSPICION ALLOCATOR — `suspicion_score(claim) -> (score, feature_breakdown)`.
     Verification is costly; the skeptic's only real skill is WHERE to point it.
     Logistic shape with HAND-SET betas (v1 is explicitly PRE-CALIBRATION: the
     betas encode a_skeptic Q1's ordering of the features, not a fit to data.
     Calibration needs a held-out replay set of independently-checked claims,
     which we do not have yet — see `BETAS` for the per-feature rationale).

     v2 CHANGES to this half:
       * `suspicion_score(..., betas=)` accepts a FITTED beta vector. The
         hand-set `BETAS` remain the DEFAULT; promotion is a deliberate act
         (`load_fitted_betas(path)` then passing the result), never a side
         effect of a file appearing on disk.
       * the uniform lie-rate placeholder is gone: `lie_rates_from_diagnosis()`
         computes PER-STAGE rates from results/diagnosis_report.json's stage
         attribution counts, which is the table the design always wanted.

  2. ATTACK MOVES — each `move_*` function consumes ledger rows only and returns
     typed findings {move, target_key, verdict, evidence} with
     verdict in {SURVIVED, INVALIDATED, UNSUPPORTED}. SURVIVED means the attack
     ran and found nothing; INVALIDATED means the attack landed; UNSUPPORTED
     means the ledgers do not carry enough to run the attack (never silently
     scored as clean — that is the silent-drop class).

GOVERNANCE (a_skeptic Q3), enforced IN CODE by
`assert_read_only_governance()`:
  * this module imports NOTHING that can modify an audited artifact (no fixer,
    no mutations, no gym runner, no subprocess, no network);
  * inputs are opened read-only and never rewritten;
  * the only write is an APPEND to the adversarial ledger, which is not an
    audited input.
The skeptic can accuse. It cannot repair, and it cannot rewrite what it audits.

Two v2 moves need powers this governance forbids, and that tension is resolved
by SEPARATION, never by weakening the rule:

  * ORACLE_AUDIT needs to CALL the oracles it scores (norm_dump, the
    witness-replay evaluator, the diagnosis engine). Importing them here would
    blow the allow-list. So this module owns the BATTERIES (pure data: source
    text plus the expected verdict per case) and the scoring loop, and the
    oracle callables are INJECTED by the caller. The skeptic still never
    imports a thing it audits.
  * MUTATION_TEST needs to write mutant source files and run a test suite in a
    subprocess. That lives entirely in `detreason/skeptic_calibration.py`, a
    separate and explicitly write-capable auditor module. It is not importable
    from here and this file's governance check is unchanged and still enforced.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
from dataclasses import dataclass, field, asdict

# --- governance: the complete allow-list of module imports for this file. Any
# import outside this set fails assert_read_only_governance(). Deliberately
# excludes fixer/mutations/gym/subprocess/urllib/shutil/os.remove-style powers.
_ALLOWED_IMPORTS = frozenset(
    # `sys` is allowed only for argv in the __main__ driver; it grants no
    # mutation power the other allowed modules do not already have.
    {"__future__", "ast", "json", "math", "os", "random", "re", "dataclasses",
     "sys"}
)
_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "fixer", "mutation", "gym", "subprocess", "urllib", "requests", "shutil",
    "socket", "http",
)

SURVIVED, INVALIDATED, UNSUPPORTED = "SURVIVED", "INVALIDATED", "UNSUPPORTED"

# Verdicts that ASSERT a repair (a_skeptic Q1 `stop`: accepting them ends work).
FIX_CLAIM_VERDICTS = frozenset({"FIX_PROPOSED", "ACCEPT"})
WEAK_CLAIM_VERDICTS = frozenset({"LOW_CONFIDENCE_PROPOSAL"})
NON_STOPPING_VERDICTS = frozenset({"ABSTAIN", "ESCALATE"})


# --------------------------------------------------------------------- claims
@dataclass
class Claim:
    """One normalized ledger row. `raw` is preserved verbatim: an aggregation
    that discards its inputs cannot be provenance-walked."""
    ledger: str
    line_no: int
    key: str
    fn: str | None
    kind: str
    verdict: str
    fixed: int | None
    gold_contact: int | None
    margins: list
    stages: list
    raw: dict

    def id(self) -> str:
        return f"{self.ledger}#{self.line_no}:{self.key}"

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_row(ledger: str, line_no: int, row: dict) -> Claim:
    """Normalize the two ledger dialects we have.

    unified_gym.jsonl : {kind,key,fn,verdict,margins,stages,fixed,gold_contact}
    real_repo_baseline_v3.jsonl : {path,fn,cls,score{...},rel_atoms,file_pick}
    A row with no verdict field is RUNNER_MEASURED: nothing model-authored
    claimed anything, the runner just scored it.
    """
    if "verdict" in row:
        return Claim(
            ledger=ledger, line_no=line_no, key=str(row.get("key")),
            fn=row.get("fn"), kind=str(row.get("kind") or "unknown"),
            verdict=str(row.get("verdict")), fixed=row.get("fixed"),
            gold_contact=row.get("gold_contact"),
            margins=list(row.get("margins") or []),
            stages=list(row.get("stages") or []), raw=row,
        )
    score = dict(row.get("score") or {})
    stages = [{"score_" + k: v} for k, v in score.items()]
    if row.get("file_pick"):
        stages.append({"file": row["file_pick"]})
    if row.get("func_pick"):
        stages.append({"function": row["func_pick"]})
    if row.get("cls"):
        stages.insert(0, {"class": row["cls"], "class_by": "runner"})
    return Claim(
        ledger=ledger, line_no=line_no, key=str(row.get("path")),
        fn=row.get("fn"), kind="real_repo", verdict="RUNNER_MEASURED",
        fixed=score.get("fixed"), gold_contact=score.get("file_hit"),
        margins=[], stages=stages, raw=row,
    )


def _boot_rows(path: str, text: str) -> list[Claim]:
    """`gauntlet_v5_boots.txt` is a summary artifact, not a per-task ledger: N
    lines each ASSERTING a total. It exists to be provenance-walked."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = re.search(r"boot\s+(\d+):\s*(\S+)\s+TOTAL\s+(\d+)\s*/\s*(\d+)", line)
        if not m:
            continue
        row = {"boot": int(m.group(1)), "suite": m.group(2),
               "total": int(m.group(3)), "denominator": int(m.group(4)),
               "line": line.strip()}
        out.append(Claim(ledger=os.path.basename(path), line_no=i,
                         key=f"{m.group(2)}#boot{m.group(1)}", fn=None,
                         kind="summary", verdict="SUMMARY_CLAIM", fixed=None,
                         gold_contact=None, margins=[], stages=[], raw=row))
    return out


def load_claims(ledger_paths) -> list[Claim]:
    """READ-ONLY consumption. Malformed lines become UNPARSEABLE claims rather
    than vanishing — a dropped input is exactly the failure this engine hunts."""
    claims: list[Claim] = []
    for p in ledger_paths:
        name = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as fh:      # mode 'r': never 'w'/'a'
            text = fh.read()
        if not p.endswith(".jsonl"):
            claims.extend(_boot_rows(p, text))
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                claims.append(Claim(name, i, f"<unparseable:{i}>", None,
                                    "unparseable", "UNPARSEABLE", None, None,
                                    [], [], {"error": str(exc)[:120]}))
                continue
            if not isinstance(row, dict):
                claims.append(Claim(name, i, f"<non-object:{i}>", None,
                                    "unparseable", "UNPARSEABLE", None, None,
                                    [], [], {"value": str(row)[:120]}))
                continue
            claims.append(_norm_row(name, i, row))
    return claims


# --------------------------------------------------------- suspicion allocator
# a_skeptic Q1: logit(S) = b0 + b_source + b_stop + b_single + b_green
#                              + b_surprise + b_lierate
# The betas below are HAND-SET to encode the ORDERING the doctrine argues for,
# not fitted. Documented rationale per beta; every one is a hypothesis awaiting
# a held-out replay set of independently re-checked claims.
BETAS = {
    # A model-authored verdict that no runner confirmed is the single strongest
    # signal we have; runner-measured facts start much lower.
    "source": 1.8,
    # Convenience: the conclusion that lets us stop working buys double scrutiny.
    "stop": 1.6,
    # No margins / one menu = one source. N repetitions from one stateless
    # source are ONE claim, so single-source-ness is not cheap to discount.
    "single": 0.9,
    # Success asserted with no discriminating evidence behind it.
    "green": 1.0,
    # Surprise vs the base rate computed FROM the ledgers at init.
    "surprise": 1.2,
    # Historical invalidation rate of the producing stage. v2: no longer a
    # uniform placeholder — `lie_rates_from_diagnosis()` fills the table from
    # the diagnosis engine's stage attribution counts.
    "lierate": 0.6,
}
BETA0 = -2.2  # base rate: most rows survive a check; suspicion must be earned.

# Features whose VALUE is derived from the gold label (`fixed`). They are
# legitimate at scoring time — the allocator scores rows AFTER a runner has or
# has not confirmed them — but they are LABEL LEAKAGE in a supervised fit and
# `skeptic_calibration.fit()` must exclude them. Named here so both halves read
# one authority instead of two hand-copied lists.
LABEL_DERIVED_FEATURES = frozenset({"green", "surprise"})
FITTABLE_FEATURES = tuple(k for k in BETAS if k not in LABEL_DERIVED_FEATURES)

FITTED_BETAS_SCHEMA = "detreason.skeptic.fitted_betas/v1"


@dataclass
class Priors:
    """Base rates computed FROM the ledgers at init (a_skeptic Q1 `surprise`)."""
    fix_rate_by_kind: dict = field(default_factory=dict)
    fix_rate_overall: float = 0.0
    # stage -> historical invalidation rate, from the diagnosis engine's stage
    # attribution counts (v2). Empty means "no diagnosis report was supplied",
    # in which case every stage falls back to `default_lie_rate` — reported as
    # such, never silently passed off as a measurement.
    lie_rate_by_stage: dict = field(default_factory=dict)
    default_lie_rate: float = 0.5
    lie_rate_source: str = "uniform-default"
    lie_rate_meta: dict = field(default_factory=dict)
    n_claims: int = 0

    def fix_rate(self, kind: str) -> float:
        return self.fix_rate_by_kind.get(kind, self.fix_rate_overall)

    def lie_rate(self, stage: str) -> float:
        if stage in self.lie_rate_by_stage:
            return self.lie_rate_by_stage[stage]
        # `_stage_name()` emits `fix:<selector>` (e.g. "fix:mutation_menu"),
        # which is the fix-select stage under a more specific name. Folding it
        # here rather than enumerating every selector keeps a NEW selector from
        # silently falling through to the default — the whole point of the
        # table is that a stage's history is known, not assumed.
        if stage.startswith("fix:") and "fix-select" in self.lie_rate_by_stage:
            return self.lie_rate_by_stage["fix-select"]
        return self.lie_rate_by_stage.get(stage, self.default_lie_rate)


# ------------------------------------------------------ per-stage lie rates (v2)
# The diagnosis engine attributes every failing row to the STAGE that broke, and
# leaves the rows that worked in the `success` cluster (stage "none"). That is
# exactly a per-stage lie-rate table:
#
#     lie_rate(S) = failures_attributed_to_S / (failures_attributed_to_S + n_success)
#
# The denominator is the right one because EVERY successful row passed through
# EVERY stage without breaking, so each success is a non-failure trial for each
# stage. Shrunk toward the corpus-wide failure rate with a Laplace-style
# pseudo-count so a stage with 1 attributed row does not read as gospel.
LIE_RATE_SHRINKAGE_K = 2.0   # pseudo-trials of the global rate mixed in

# The complete stage vocabulary the diagnosis engine can emit. A stage in this
# set that has NO cluster in a given report means "observed zero failures",
# which is a MEASUREMENT — not "unknown", which is the default fallback. The
# two are different facts and conflating them silently reverts a measured stage
# to the uniform placeholder the moment a corpus stops producing that failure.
# (Measured 2026-08-28: the `abstained` cluster went to zero between two gym
# runs and every abstain-stage row silently fell back to the default.)
_DIAGNOSIS_STAGES = ("locate", "function-select", "fix-select", "abstain",
                     "none")

# skeptic `_stage_name()` vocabulary -> diagnosis stage vocabulary. Kept
# explicit rather than fuzzy-matched: an unmapped name falls back to
# `default_lie_rate` and says so, which is the fail-loud direction.
_STAGE_ALIASES = {
    "proposal": "fix-select",
    "abstain": "abstain",
    "escalate": "abstain",
    "low_confidence_proposal": "fix-select",
    "fix_proposed": "fix-select",
    "accept": "fix-select",
    "runner_measured": "locate",
}


def lie_rates_from_diagnosis(report_path: str):
    """(table, meta) from a diagnosis_report.json. READ-ONLY.

    `table` is keyed by BOTH the diagnosis stage names and the skeptic's own
    `_stage_name()` vocabulary (via `_STAGE_ALIASES`, plus every `fix:<x>`
    stage folded onto `fix-select`), so `Priors.lie_rate()` resolves without a
    second translation layer at the call site.
    """
    with open(report_path, "r", encoding="utf-8") as fh:   # mode 'r': never 'w'
        report = json.load(fh)
    clusters = report.get("clusters") or {}
    per_stage: dict = {}
    n_success = 0
    for cluster in clusters.values():
        if not isinstance(cluster, dict):
            continue
        stage = str(cluster.get("stage") or "unknown")
        count = int(cluster.get("count") or 0)
        if stage == "none":                 # the diagnosis engine's success bucket
            n_success += count
            continue
        per_stage[stage] = per_stage.get(stage, 0) + count
    n_failure = sum(per_stage.values())
    total = n_failure + n_success
    global_rate = (n_failure / total) if total else 0.5

    table: dict = {}
    raw: dict = {}
    for stage, fails in sorted(per_stage.items()):
        trials = fails + n_success
        rate = ((fails + LIE_RATE_SHRINKAGE_K * global_rate)
                / (trials + LIE_RATE_SHRINKAGE_K)) if trials else global_rate
        table[stage] = round(rate, 6)
        raw[stage] = {"failures": fails, "successes": n_success,
                      "trials": trials,
                      "unshrunk": round(fails / trials, 6) if trials else None}
    # Known stages with no cluster in THIS report observed zero failures over
    # `n_success` trials. That is a measured near-zero rate, not an unknown.
    if n_success:
        zero_rate = round((LIE_RATE_SHRINKAGE_K * global_rate)
                          / (n_success + LIE_RATE_SHRINKAGE_K), 6)
        for stage in _DIAGNOSIS_STAGES:
            if stage in table:
                continue
            table[stage] = zero_rate
            raw[stage] = {"failures": 0, "successes": n_success,
                          "trials": n_success, "unshrunk": 0.0,
                          "note": "known stage with no cluster in this report: "
                                  "zero failures OBSERVED, not unknown"}

    for alias, stage in _STAGE_ALIASES.items():
        if stage in table:
            table[alias] = table[stage]
    meta = {"source": os.path.basename(report_path),
            "n_failure_rows": n_failure, "n_success_rows": n_success,
            "global_failure_rate": round(global_rate, 6),
            "shrinkage_k": LIE_RATE_SHRINKAGE_K,
            "per_stage_counts": raw,
            "rows_attributed": report.get("rows_attributed")}
    return table, meta


def build_priors(claims, diagnosis_report_path: str | None = None) -> Priors:
    by_kind: dict = {}
    n_fix = n_tot = 0
    for c in claims:
        if c.fixed is None:
            continue
        agg = by_kind.setdefault(c.kind, [0, 0])
        agg[1] += 1
        n_tot += 1
        if c.fixed == 1:
            agg[0] += 1
            n_fix += 1
    table, meta, source = {}, {}, "uniform-default"
    if diagnosis_report_path and os.path.exists(diagnosis_report_path):
        table, meta = lie_rates_from_diagnosis(diagnosis_report_path)
        source = f"diagnosis:{meta['source']}"
    return Priors(
        fix_rate_by_kind={k: (v[0] / v[1] if v[1] else 0.0)
                          for k, v in by_kind.items()},
        fix_rate_overall=(n_fix / n_tot) if n_tot else 0.0,
        lie_rate_by_stage=table,
        lie_rate_source=source,
        lie_rate_meta=meta,
        n_claims=len(claims),
    )


# ------------------------------------------------------ fitted betas (opt-in)
def load_fitted_betas(path: str) -> dict:
    """Load a fitted beta vector written by `skeptic_calibration.fit()`.

    Returns the beta dict. It is NOT installed anywhere: the caller must hand
    it to `suspicion_score(..., betas=)` / `audit(..., betas=)`. Promotion of a
    fitted model over the hand-set doctrine betas is a deliberate act, and a
    file appearing on disk is not one.

    Fails LOUD on a bad file — a calibration that silently falls back to the
    hand-set betas while reporting "fitted" is the exact silent-drop class this
    engine exists to hunt.
    """
    with open(path, "r", encoding="utf-8") as fh:          # mode 'r': never 'w'
        blob = json.load(fh)
    if not isinstance(blob, dict):
        raise ValueError(f"fitted betas file is not an object: {path}")
    if blob.get("schema") != FITTED_BETAS_SCHEMA:
        raise ValueError(
            f"fitted betas schema mismatch: expected {FITTED_BETAS_SCHEMA}, "
            f"got {blob.get('schema')!r}")
    betas = blob.get("betas")
    if not isinstance(betas, dict):
        raise ValueError(f"fitted betas file carries no 'betas' object: {path}")
    missing = sorted(set(BETAS) - set(betas))
    if missing:
        raise ValueError(f"fitted betas are missing features {missing}")
    extra = sorted(set(betas) - set(BETAS))
    if extra:
        raise ValueError(f"fitted betas carry unknown features {extra}")
    out = {k: float(betas[k]) for k in BETAS}
    if "beta0" in blob:
        out["_beta0"] = float(blob["beta0"])
    return out


def _as_claim(claim) -> Claim:
    if isinstance(claim, Claim):
        return claim
    if isinstance(claim, dict):
        if "ledger" in claim and "raw" in claim:
            return Claim(**{k: claim.get(k) for k in Claim.__dataclass_fields__})
        return _norm_row(claim.get("_ledger", "<inline>"), 0, claim)
    raise TypeError(f"suspicion_score expects a Claim or dict, got {type(claim)}")


def _stage_name(c: Claim) -> str:
    """The producing stage, for the lie-rate table."""
    for st in c.stages:
        if isinstance(st, dict) and "fix" in st:
            return f"fix:{st['fix']}"
    if c.verdict in FIX_CLAIM_VERDICTS or c.verdict in WEAK_CLAIM_VERDICTS:
        return "proposal"
    return c.verdict.lower()


def suspicion_score(claim, priors: Priors | None = None, betas: dict | None = None):
    """S(x) = P(an independent check invalidates x).

    Returns (score in [0,1], feature_breakdown). The breakdown carries the raw
    feature values, the betas, and each contribution, so a suspicion number can
    itself be provenance-walked — the allocator is not exempt from its own rule.

    `betas` defaults to the HAND-SET doctrine vector. Pass the dict returned by
    `load_fitted_betas()` to score with a fitted model instead; the breakdown's
    `calibration` field records which one was actually used, so no report can
    claim "fitted" while running on the defaults.
    """
    c = _as_claim(claim)
    pri = priors or Priors()

    claims_fix = c.verdict in FIX_CLAIM_VERDICTS
    weak_claim = c.verdict in WEAK_CLAIM_VERDICTS
    model_authored = claims_fix or weak_claim
    runner_confirmed = (c.fixed == 1)
    gold = (c.gold_contact == 1)

    # source: model-authored and unconfirmed is the top of the scale; a
    # model-authored claim a runner later confirmed drops most of the way;
    # a pure runner measurement starts near zero.
    if model_authored and not runner_confirmed:
        f_source = 1.0
    elif model_authored:
        f_source = 0.3
    elif c.verdict == "UNPARSEABLE":
        f_source = 1.0            # we could not even read it: assume the worst
    else:
        f_source = 0.0

    # stop-convenience: does accepting this end the work?
    if c.verdict in FIX_CLAIM_VERDICTS:
        f_stop = 1.0
    elif weak_claim:
        f_stop = 0.6
    elif c.verdict == "SUMMARY_CLAIM":
        f_stop = 0.5              # a summary closes a question by construction
    elif c.verdict in NON_STOPPING_VERDICTS:
        f_stop = 0.1              # ABSTAIN keeps the work open: least convenient
    else:
        f_stop = 0.3

    # single-source: no margins recorded at all, or a single menu, or every
    # recorded margin is null (a menu that reports no margin is not a second
    # observation, it is the same observation with a confidence field).
    menus = [m for m in c.margins if isinstance(m, dict)]
    real_margins = [m for m in menus if m.get("margin") is not None]
    if not menus or len(menus) <= 1:
        f_single = 1.0
    elif not real_margins:
        f_single = 0.8
    elif len(real_margins) < len(menus):
        f_single = 0.5
    else:
        f_single = 0.2

    # green-without-discrimination: success asserted with nothing behind it that
    # could have said otherwise. Two shapes, both "a claim of success whose only
    # evidence is the claim": (a) fixed=1 with no independent gold contact,
    # (b) a fix asserted that the runner never confirmed at all.
    if runner_confirmed and not gold:
        f_green = 1.0
    elif runner_confirmed and not _has_gate_stage(c):
        f_green = 0.5             # gold contact is real evidence; still no gate
    elif claims_fix and not runner_confirmed:
        f_green = 1.0 if not gold else 0.7
    else:
        f_green = 0.0

    # surprise: -log-prob-ish under the base rate computed from the ledgers.
    # A fix on a family that almost never fixes is surprising; gold contact is
    # independent corroboration, so it damps the surprise by half.
    base = pri.fix_rate(c.kind)
    if c.fixed == 1:
        f_surprise = 1.0 - base
        if gold:
            f_surprise *= 0.5
    elif c.fixed == 0:
        f_surprise = base
    else:
        f_surprise = 0.0

    f_lierate = pri.lie_rate(_stage_name(c))

    feats = {"source": f_source, "stop": f_stop, "single": f_single,
             "green": f_green, "surprise": round(f_surprise, 4),
             "lierate": f_lierate}
    if betas is None:
        b, b0, tag = BETAS, BETA0, "v1-heuristic-preCalibration"
    else:
        b = betas
        b0 = betas.get("_beta0", BETA0)
        tag = "v2-fitted"
    contrib = {k: round(b[k] * v, 4) for k, v in feats.items()}
    logit = b0 + sum(contrib.values())
    score = 1.0 / (1.0 + math.exp(-logit))
    breakdown = {
        "features": feats,
        "betas": {k: b[k] for k in feats}, "beta0": b0,
        "contributions": contrib, "logit": round(logit, 4),
        "stage": _stage_name(c), "kind": c.kind, "verdict": c.verdict,
        "base_fix_rate": round(base, 4),
        "lie_rate_source": pri.lie_rate_source,
        "calibration": tag,
    }
    return round(score, 6), breakdown


def _has_gate_stage(c: Claim) -> bool:
    """Did anything beyond 'it compiled' discriminate here? v1 looks for an
    explicit gate/test/verify stage; absence is reported, never assumed clean."""
    for st in c.stages:
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            blob = f"{k}:{v}".lower()
            if any(w in blob for w in ("gate", "test", "verify", "assert",
                                       "oracle", "gold")):
                return True
    return False


# ---------------------------------------------------------------- attack moves
def _finding(move, target_key, verdict, evidence) -> dict:
    return {"move": move, "target_key": target_key, "verdict": verdict,
            "evidence": evidence}


def _claims_fix_bool(c: Claim) -> bool:
    return c.verdict in FIX_CLAIM_VERDICTS


def claim_replication_check(claims) -> list[dict]:
    """Move 1 (a_skeptic Q2 `claim_replication`). Two ledger rows for the SAME
    key that disagree on fixed/verdict is a replication failure: at most one of
    them can be true, and neither ledger knows it."""
    by_key: dict = {}
    for c in claims:
        if c.kind in ("unparseable", "summary"):
            continue
        by_key.setdefault(c.key, []).append(c)
    out = []
    for key, rows in sorted(by_key.items()):
        ledgers = {r.ledger for r in rows}
        if len(ledgers) < 2:
            out.append(_finding(
                "claim_replication_check", key, UNSUPPORTED,
                {"reason": "key appears in a single ledger; nothing to "
                           "replicate against",
                 "ledgers": sorted(ledgers), "n_rows": len(rows)}))
            continue
        fixed_vals = {r.fixed for r in rows if r.fixed is not None}
        disagree_fixed = len(fixed_vals) > 1
        # A fix ASSERTED but never gold-scored in one ledger, while ANOTHER
        # ledger measured the same key unfixed. Scored-vs-scored disagreement is
        # already `disagree_fixed`; a row whose own ledger scored it fixed=0 is
        # a convenience failure, not a replication failure — that belongs to
        # convenience_inversion and must not be double-counted here.
        unscored_claim_ledgers = {r.ledger for r in rows
                                  if r.fixed is None and _claims_fix_bool(r)}
        measured_unfixed_ledgers = {r.ledger for r in rows if r.fixed == 0}
        disagree_claim = bool(unscored_claim_ledgers - measured_unfixed_ledgers) \
            and bool(measured_unfixed_ledgers)
        if disagree_fixed or disagree_claim:
            out.append(_finding(
                "claim_replication_check", key, INVALIDATED,
                {"reason": "same key, different outcome across ledgers",
                 "fixed_values": sorted(v for v in fixed_vals),
                 "rows": [{"ledger": r.ledger, "line": r.line_no,
                           "verdict": r.verdict, "fixed": r.fixed,
                           "gold_contact": r.gold_contact, "fn": r.fn}
                          for r in rows]}))
        else:
            out.append(_finding(
                "claim_replication_check", key, SURVIVED,
                {"reason": "outcome agrees across ledgers",
                 "ledgers": sorted(ledgers),
                 "fixed_values": sorted(v for v in fixed_vals)}))
    return out


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2}


def _stage_signature(c: Claim) -> str:
    """The behavioural fingerprint of a run: what it classified and what it
    picked. Two DIFFERENT tasks producing the identical signature means the
    inputs did not reach the decision (a counterfactual that should have moved
    the answer and did not)."""
    parts = []
    for st in c.stages:
        if not isinstance(st, dict):
            continue
        for k in ("class", "class_by", "file", "function", "mutations"):
            if k in st:
                parts.append(f"{k}={st[k]}")
    for m in c.margins:
        if isinstance(m, dict):
            parts.append(f"pick[{m.get('menu')}]={m.get('pick')}")
    return "|".join(parts)


def counterfactual_consistency(claims) -> list[dict]:
    """Move 2 (a_skeptic Q2 `counterfactual_probe`). Two checks:
    (a) constant behaviour — identical class_by/menu picks across DIFFERENT
        tasks, i.e. the task input changed and the answer did not;
    (b) evidence linkage — a FIX_PROPOSED whose 'change' text shares zero tokens
        with its own task (key + fn). Evidence linkage should have caught it.
    """
    out = []
    by_sig: dict = {}
    for c in claims:
        if c.kind in ("unparseable", "summary") or not c.stages:
            continue
        sig = _stage_signature(c)
        if sig:
            by_sig.setdefault((c.ledger, sig), []).append(c)
    for (ledger, sig), rows in sorted(by_sig.items(), key=lambda kv: str(kv[0])):
        distinct_tasks = {(r.key, r.fn) for r in rows}
        if len(distinct_tasks) > 1:
            out.append(_finding(
                "counterfactual_consistency",
                ",".join(sorted(k for k, _ in distinct_tasks)), INVALIDATED,
                {"check": "constant_behavior",
                 "reason": "identical stage picks across different tasks: the "
                           "input changed and the decision did not",
                 "ledger": ledger, "signature": sig[:400],
                 "tasks": sorted(f"{k}::{f}" for k, f in distinct_tasks)}))

    for c in claims:
        if not _claims_fix_bool(c):
            continue
        change = ""
        for st in c.stages:
            if isinstance(st, dict) and "change" in st:
                change = str(st["change"])
                break
        if not change:
            out.append(_finding(
                "counterfactual_consistency", c.key, UNSUPPORTED,
                {"check": "evidence_linkage",
                 "reason": "fix claim carries no 'change' stage text to link",
                 "ledger": c.ledger, "line": c.line_no}))
            continue
        task_tokens = _tokens(c.key) | _tokens(c.fn or "")
        shared = task_tokens & _tokens(change)
        if not shared:
            out.append(_finding(
                "counterfactual_consistency", c.key, INVALIDATED,
                {"check": "evidence_linkage",
                 "reason": "fix text shares zero tokens with its own task; "
                           "evidence linkage should have rejected it",
                 "ledger": c.ledger, "line": c.line_no, "fn": c.fn,
                 "change": change[:300]}))
        else:
            out.append(_finding(
                "counterfactual_consistency", c.key, SURVIVED,
                {"check": "evidence_linkage", "shared_tokens": sorted(shared),
                 "ledger": c.ledger, "line": c.line_no}))
    return out


def convenience_inversion(claims) -> list[dict]:
    """Move 3 (a_skeptic Q2 `convenience_inversion`). Every claim that ENDS the
    work (a proposed fix) and was gold-scored as not fixed is a confidently
    wrong instance. Enumerate them and cluster by the stage evidence, because
    the cluster is the defect, not the instance."""
    out = []
    clusters: dict = {}
    for c in claims:
        if not _claims_fix_bool(c):
            continue
        if c.fixed is None:
            out.append(_finding(
                "convenience_inversion", c.key, UNSUPPORTED,
                {"reason": "fix claim in a ledger with no gold scoring",
                 "ledger": c.ledger, "line": c.line_no, "verdict": c.verdict}))
            continue
        if c.fixed == 0:
            cls = "unknown"
            fix_stage = "unknown"
            for st in c.stages:
                if isinstance(st, dict):
                    if "class" in st:
                        cls = str(st["class"])
                    if "fix" in st:
                        fix_stage = str(st["fix"])
            cluster_key = f"{cls}/{fix_stage}"
            clusters.setdefault(cluster_key, []).append(c.id())
            out.append(_finding(
                "convenience_inversion", c.key, INVALIDATED,
                {"reason": "verdict asserts a fix; gold scoring says fixed=0",
                 "ledger": c.ledger, "line": c.line_no, "verdict": c.verdict,
                 "fn": c.fn, "gold_contact": c.gold_contact,
                 "cluster": cluster_key}))
        else:
            out.append(_finding(
                "convenience_inversion", c.key, SURVIVED,
                {"reason": "fix claim confirmed by gold scoring (fixed=1)",
                 "ledger": c.ledger, "line": c.line_no,
                 "gold_contact": c.gold_contact}))
    if clusters:
        out.append(_finding(
            "convenience_inversion", "<clusters>", INVALIDATED,
            {"reason": "confidently-wrong fix claims clustered by "
                       "class/fix-stage",
             "clusters": {k: sorted(v) for k, v in sorted(clusters.items())},
             "n_instances": sum(len(v) for v in clusters.values())}))
    return out


def provenance_walk(claims) -> list[dict]:
    """Move 4 (a_skeptic Q2 `provenance_walk`). Every aggregate we might print
    is recomputed from the raw rows, and any summary row in the inputs is
    checked AGAINST that recomputation. A summary nobody can reconstruct is
    UNSUPPORTED, never quoted."""
    out = []
    per_ledger: dict = {}
    for c in claims:
        if c.kind in ("summary", "unparseable"):
            continue
        agg = per_ledger.setdefault(c.ledger, {
            "rows": 0, "fixed_1": 0, "fixed_0": 0, "fixed_null": 0,
            "gold_contact_1": 0, "fix_claims": 0, "verdicts": {}})
        agg["rows"] += 1
        agg["verdicts"][c.verdict] = agg["verdicts"].get(c.verdict, 0) + 1
        if c.fixed == 1:
            agg["fixed_1"] += 1
        elif c.fixed == 0:
            agg["fixed_0"] += 1
        else:
            agg["fixed_null"] += 1
        if c.gold_contact == 1:
            agg["gold_contact_1"] += 1
        if _claims_fix_bool(c):
            agg["fix_claims"] += 1
    for ledger, agg in sorted(per_ledger.items()):
        consistent = (agg["fixed_1"] + agg["fixed_0"] + agg["fixed_null"]
                      == agg["rows"])
        out.append(_finding(
            "provenance_walk", ledger, SURVIVED if consistent else INVALIDATED,
            {"check": "recomputed_from_raw_rows", "totals": agg,
             "reason": "counts partition the row set" if consistent
                       else "counts do not partition the row set"}))

    summaries = [c for c in claims if c.kind == "summary"]
    if summaries:
        totals = {(c.raw.get("suite"), c.raw.get("total"),
                   c.raw.get("denominator")) for c in summaries}
        if len(totals) > 1:
            out.append(_finding(
                "provenance_walk", "boot_summaries", INVALIDATED,
                {"check": "summary_rows_disagree",
                 "distinct_totals": sorted(str(t) for t in totals),
                 "n_summaries": len(summaries)}))
        else:
            suite, total, denom = next(iter(totals))
            out.append(_finding(
                "provenance_walk", str(suite), UNSUPPORTED,
                {"check": "summary_has_no_raw_rows_in_scope",
                 "reason": "every boot asserts the same total, but no per-case "
                           "rows for this suite are in the audited ledgers, so "
                           "the number cannot be recomputed — identical "
                           "repetitions from one runner are ONE claim",
                 "asserted": f"{total}/{denom}", "n_repetitions": len(summaries)}))
    unparseable = [c for c in claims if c.kind == "unparseable"]
    if unparseable:
        out.append(_finding(
            "provenance_walk", "<unparseable_rows>", INVALIDATED,
            {"check": "input_completeness",
             "reason": "ledger lines could not be parsed; any aggregate over "
                       "this ledger is incomplete",
             "rows": [c.id() for c in unparseable]}))
    return out


def second_body_sweep(claims) -> list[dict]:
    """Move 5 (a_skeptic Q2 `second_body_sweep`). For keys that failed in EVERY
    ledger that saw them, mask nothing and ask: is the failure attribution a
    single cause, or several independent stage failures? A single-cause story
    over a multi-cause corpse is how the second body stays buried."""
    by_key: dict = {}
    for c in claims:
        if c.kind in ("unparseable", "summary"):
            continue
        by_key.setdefault(c.key, []).append(c)
    out = []
    for key, rows in sorted(by_key.items()):
        scored = [r for r in rows if r.fixed is not None]
        if not scored:
            out.append(_finding(
                "second_body_sweep", key, UNSUPPORTED,
                {"reason": "no gold-scored row for this key"}))
            continue
        if any(r.fixed == 1 for r in scored):
            out.append(_finding(
                "second_body_sweep", key, SURVIVED,
                {"reason": "key succeeded in at least one ledger; not a "
                           "universal failure",
                 "ledgers": sorted({r.ledger for r in scored})}))
            continue
        causes = set()
        detail = []
        for r in scored:
            if r.gold_contact != 1:
                causes.add("no_gold_contact")
            sc = r.raw.get("score") if isinstance(r.raw.get("score"), dict) else {}
            for stage_name, flag in (("file_miss", "file_hit"),
                                     ("function_miss", "func_hit"),
                                     ("file_not_in_menu", "file_in_menu"),
                                     ("mutation_not_in_menu", "mut_in_menu")):
                if flag in sc and not sc[flag]:
                    causes.add(stage_name)
            if r.verdict in NON_STOPPING_VERDICTS:
                causes.add("abstained")
            if _claims_fix_bool(r):
                causes.add("wrong_fix_proposed")
            if r.verdict in WEAK_CLAIM_VERDICTS:
                causes.add("low_confidence_proposal")
            detail.append({"ledger": r.ledger, "line": r.line_no,
                           "verdict": r.verdict, "gold_contact": r.gold_contact,
                           "score": sc or None})
        out.append(_finding(
            "second_body_sweep", key, INVALIDATED,
            {"reason": "failed in every ledger that scored it",
             "attribution": "single_cause" if len(causes) <= 1
                            else "multiple_independent_stage_failures",
             "causes": sorted(causes), "n_causes": len(causes),
             "rows": detail}))
    return out


# ------------------------------------------------------- oracle audit (move 6)
# a_skeptic Q2 `oracle_audit(oracle, known_cases) -> OracleCalibrationCertificate`:
# "The oracle is audited before blaming the model."
#
# The batteries below are PURE DATA. Every case names the oracle, the inputs,
# and the expected answer, and — critically — the batteries contain cases that
# must hold AND cases that must NOT hold. A battery of only-positives cannot
# distinguish a correct oracle from one that says yes to everything, which is
# the vacuity failure this whole engine exists to hunt.
#
# The oracle callables themselves are INJECTED (see the module docstring: this
# module may not import what it audits).

# (a) norm_dump: behavioral equivalence of two function sources.
#     Adapter contract: equivalent(src_a: str, src_b: str) -> bool
NORM_DUMP_BATTERY = (
    {"case": "mirrored_comparison_is_equivalent",
     "a": "def f(a, b):\n    return a < b\n",
     "b": "def f(a, b):\n    return b > a\n",
     "expect": True,
     "why": "a<b and b>a are the same predicate; the v2 oracle normalizes "
            "operand order, so textual identity would have scored a correct "
            "fix as wrong"},
    {"case": "boolop_operand_order_is_equivalent",
     "a": "def f(x, y):\n    return x and y\n",
     "b": "def f(x, y):\n    return y and x\n",
     "expect": True,
     "why": "and/or operands are sorted canonically"},
    {"case": "mirrored_gte_is_equivalent",
     "a": "def f(a, b):\n    if a >= b:\n        return 1\n    return 0\n",
     "b": "def f(a, b):\n    if b <= a:\n        return 1\n    return 0\n",
     "expect": True,
     "why": ">= mirrors to <= with the operands swapped"},
    {"case": "reversed_comparison_is_NOT_equivalent",
     "a": "def f(a, b):\n    return a < b\n",
     "b": "def f(a, b):\n    return a > b\n",
     "expect": False,
     "why": "the direction genuinely reversed: this is the bug, not a mirror"},
    {"case": "and_vs_or_is_NOT_equivalent",
     "a": "def f(x, y):\n    return x and y\n",
     "b": "def f(x, y):\n    return x or y\n",
     "expect": False,
     "why": "different connective, different behavior"},
    {"case": "boundary_tweak_is_NOT_equivalent",
     "a": "def f(a, b):\n    return a < b\n",
     "b": "def f(a, b):\n    return a <= b\n",
     "expect": False,
     "why": "< vs <= differ on equality; an oracle that folds these would "
            "credit an off-by-one as a fix"},
    {"case": "none_identity_flip_is_NOT_equivalent",
     "a": "def f(x):\n    return x is None\n",
     "b": "def f(x):\n    return x is not None\n",
     "expect": False,
     "why": "the None check is inverted"},
    {"case": "different_variable_is_NOT_equivalent",
     "a": "def f(a, b):\n    return a < b\n",
     "b": "def f(a, b):\n    return a < a\n",
     "expect": False,
     "why": "wrong_variable family: same shape, different operand"},
)

# (b) witness_replay's safe evaluator: known truth tables.
#     Adapter contract: evaluate(expr_src: str, env: dict) -> ("OK", repr) |
#                       ("ERR", exception_name) | ("UNSUPPORTED", reason)
SAFE_EVAL_BATTERY = (
    {"case": "and_true_true", "expr": "a and b",
     "env": {"a": True, "b": True}, "expect": ("OK", "True")},
    {"case": "and_true_false", "expr": "a and b",
     "env": {"a": True, "b": False}, "expect": ("OK", "False")},
    {"case": "or_false_true", "expr": "a or b",
     "env": {"a": False, "b": True}, "expect": ("OK", "True")},
    {"case": "or_false_false", "expr": "a or b",
     "env": {"a": False, "b": False}, "expect": ("OK", "False")},
    {"case": "not_true", "expr": "not a", "env": {"a": True},
     "expect": ("OK", "False")},
    {"case": "lt_orders_ints", "expr": "a < b", "env": {"a": 0, "b": 1},
     "expect": ("OK", "True")},
    {"case": "gt_orders_ints", "expr": "a > b", "env": {"a": 0, "b": 1},
     "expect": ("OK", "False")},
    {"case": "is_none_true", "expr": "a is None", "env": {"a": None},
     "expect": ("OK", "True")},
    {"case": "is_not_none_on_zero", "expr": "a is not None", "env": {"a": 0},
     "expect": ("OK", "True"),
     "why": "0 is falsey but NOT None — conflating them is a real bug class"},
    {"case": "len_of_empty", "expr": "len(a) > 0", "env": {"a": []},
     "expect": ("OK", "False")},
    {"case": "len_of_nonempty", "expr": "len(a) > 0", "env": {"a": [1]},
     "expect": ("OK", "True")},
    {"case": "and_shortcircuits_to_operand", "expr": "a and b",
     "env": {"a": 0, "b": 1}, "expect": ("OK", "0"),
     "why": "python `and` returns the OPERAND, not a bool; an evaluator that "
            "coerced would erase deltas between 0 and False"},
    {"case": "none_lt_int_is_a_typeerror", "expr": "a < b",
     "env": {"a": None, "b": 1}, "expect": ("ERR", "TypeError"),
     "why": "a legitimate row outcome, recorded — never swallowed as False"},
    {"case": "dotted_attribute_binds_opaquely", "expr": "user.profile.age > 3",
     "env": {"user.profile.age": 4}, "expect": ("OK", "True")},
    {"case": "unwhitelisted_call_is_unsupported", "expr": "foo(a) > 1",
     "env": {"a": 1}, "expect": ("UNSUPPORTED", None),
     "why": "fail-open is the law: outside the evaluable subset it must say so"},
    {"case": "unbound_name_is_unsupported", "expr": "zzz > 1", "env": {},
     "expect": ("UNSUPPORTED", None)},
)

# (c) the diagnosis engine as an oracle: synthetic rows with a known stage.
#     Adapter contract: attribute(rows: list[dict]) -> dict cluster_name -> count
DIAGNOSIS_BATTERY = (
    {"case": "gold_file_never_reached_the_menu_is_locate_recall",
     "row": {"path": "x/a.py", "fn": "f", "cls": "bool_flip",
             "score": {"file_in_menu": 0, "file_hit": 0}},
     "expect_stage": "locate", "expect_cluster": "locate.recall"},
    {"case": "gold_file_in_menu_but_not_picked_is_locate_pick",
     "row": {"path": "x/b.py", "fn": "f", "cls": "bool_flip",
             "score": {"file_in_menu": 1, "file_hit": 0}},
     "expect_stage": "locate", "expect_cluster": "locate.pick"},
    {"case": "right_file_wrong_function_is_function_select",
     "row": {"path": "x/c.py", "fn": "f", "cls": "bool_flip",
             "score": {"file_in_menu": 1, "file_hit": 1, "func_hit": 0}},
     "expect_stage": "function-select", "expect_cluster": "function.pick"},
    {"case": "right_everything_wrong_fix_is_fix_select",
     "row": {"path": "x/d.py", "fn": "f", "cls": "bool_flip",
             "score": {"file_in_menu": 1, "file_hit": 1, "func_hit": 1,
                       "mut_in_menu": 1, "fixed": 0}},
     "expect_stage": "fix-select", "expect_cluster": "fix.pick.model"},
    {"case": "a_fixed_row_is_not_a_failure",
     "row": {"path": "x/e.py", "fn": "f", "cls": "bool_flip",
             "score": {"file_in_menu": 1, "file_hit": 1, "func_hit": 1,
                       "mut_in_menu": 1, "fixed": 1}},
     "expect_stage": "none", "expect_cluster": "success"},
)

ORACLE_BATTERIES = {
    "norm_dump": NORM_DUMP_BATTERY,
    "witness_replay.safe_eval": SAFE_EVAL_BATTERY,
    "diagnosis.attribute": DIAGNOSIS_BATTERY,
}


def _score_norm_dump(equivalent, battery) -> list[dict]:
    cases = []
    for spec in battery:
        try:
            got = bool(equivalent(spec["a"], spec["b"]))
            err = None
        except Exception as exc:                    # an oracle that RAISES fails
            got, err = None, f"{type(exc).__name__}: {exc}"[:200]
        cases.append({"case": spec["case"], "expected": spec["expect"],
                      "got": got, "error": err,
                      "ok": (err is None and got == spec["expect"]),
                      "why": spec.get("why", "")})
    return cases


def _score_safe_eval(evaluate, battery) -> list[dict]:
    cases = []
    for spec in battery:
        exp_kind, exp_val = spec["expect"]
        try:
            got = tuple(evaluate(spec["expr"], dict(spec["env"])))
            err = None
        except Exception as exc:
            got, err = None, f"{type(exc).__name__}: {exc}"[:200]
        if err is not None:
            ok = False
        elif exp_val is None:                # kind-only case (UNSUPPORTED)
            ok = got[0] == exp_kind
        else:
            ok = got == (exp_kind, exp_val)
        cases.append({"case": spec["case"], "expr": spec["expr"],
                      "expected": [exp_kind, exp_val],
                      "got": list(got) if got else None, "error": err,
                      "ok": ok, "why": spec.get("why", "")})
    return cases


def _score_diagnosis(attribute, battery) -> list[dict]:
    """Two properties at once: (1) each synthetic row lands in the expected
    cluster, and (2) ATTRIBUTION STABILITY — re-running the identical input
    yields the identical clustering. An oracle whose answer drifts between
    calls cannot ground any verdict."""
    cases = []
    for spec in battery:
        try:
            first = attribute([spec["row"]])
            second = attribute([spec["row"]])
            err = None
        except Exception as exc:
            first = second = None
            err = f"{type(exc).__name__}: {exc}"[:200]
        if err is not None:
            ok, got, stable = False, None, None
        else:
            got = max(first, key=lambda k: first[k]) if first else None
            stable = (first == second)
            ok = (got == spec["expect_cluster"]) and stable
        cases.append({"case": spec["case"], "expected": spec["expect_cluster"],
                      "got": got, "stable_across_reruns": stable,
                      "error": err, "ok": ok})
    return cases


_ORACLE_SCORERS = {
    "norm_dump": _score_norm_dump,
    "witness_replay.safe_eval": _score_safe_eval,
    "diagnosis.attribute": _score_diagnosis,
}


def oracle_audit(oracles: dict | None = None) -> list[dict]:
    """Move 6 (a_skeptic Q2 `oracle_audit`). Score each injected oracle against
    its known-case battery and stamp an OracleCalibrationCertificate.

    `oracles` maps an oracle name in `ORACLE_BATTERIES` to its adapter callable.
    An oracle with no adapter supplied is UNSUPPORTED — never scored clean,
    because "we did not run it" and "it passed" are different facts.

    A battery case that fails is a REAL finding about the oracle, and the
    certificate carries the per-case evidence so the failure is nameable.
    """
    oracles = oracles or {}
    out = []
    for name in sorted(ORACLE_BATTERIES):
        battery = ORACLE_BATTERIES[name]
        fn = oracles.get(name)
        if fn is None:
            out.append(_finding(
                "oracle_audit", name, UNSUPPORTED,
                {"certificate": "OracleCalibrationCertificate",
                 "reason": "no oracle adapter injected; the battery did not run",
                 "n_cases": len(battery)}))
            continue
        cases = _ORACLE_SCORERS[name](fn, battery)
        failed = [c for c in cases if not c["ok"]]
        out.append(_finding(
            "oracle_audit", name,
            SURVIVED if not failed else INVALIDATED,
            {"certificate": "OracleCalibrationCertificate",
             "n_cases": len(cases), "n_passed": len(cases) - len(failed),
             "n_failed": len(failed),
             # DISCRIMINATION, measured rather than inferred from case names:
             # how many DISTINCT answers the battery expects. A battery whose
             # cases all expect the same answer cannot tell a correct oracle
             # from a constant one, and would certify anything.
             "discrimination": {
                 "n_distinct_expected": len(
                     {json.dumps(c.get("expected"), sort_keys=True)
                      for c in cases}),
                 "expected_values": sorted(
                     {json.dumps(c.get("expected"), sort_keys=True)
                      for c in cases}),
             },
             "failed_cases": failed,
             "cases": cases,
             "reason": ("every known case scored correctly" if not failed
                        else f"{len(failed)} known case(s) scored WRONG; the "
                             "oracle is audited before the model is blamed")}))
    return out


MOVES = {
    "claim_replication_check": claim_replication_check,
    "counterfactual_consistency": counterfactual_consistency,
    "convenience_inversion": convenience_inversion,
    "provenance_walk": provenance_walk,
    "second_body_sweep": second_body_sweep,
}


# ------------------------------------------------------------------ governance
def assert_read_only_governance(module_path: str | None = None) -> dict:
    """a_skeptic Q3, enforced in code: the skeptic may accuse, never repair.

    Parses THIS module's own source and proves (a) every import is in the
    allow-list, (b) nothing imports the fixer / mutation / gym machinery or any
    process/network power, (c) the only non-read file mode used anywhere in the
    module is the append to the adversarial ledger. Raises on violation; the
    audit calls it before touching an input.
    """
    path = module_path or __file__
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    illegal = sorted(n for n in imported if n not in _ALLOWED_IMPORTS)
    if illegal:
        raise RuntimeError(
            f"skeptic governance violation: disallowed imports {illegal}")
    forbidden = sorted(n for n in imported
                       if any(s in n.lower() for s in _FORBIDDEN_IMPORT_SUBSTRINGS))
    if forbidden:
        raise RuntimeError(
            f"skeptic governance violation: mutation-capable imports {forbidden}")
    modes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            mode = "r"
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords or []:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            modes.append(mode)
    writes = [m for m in modes if m not in ("r", "rt", "rb")]
    if any(m not in ("a", "at") for m in writes):
        raise RuntimeError(
            f"skeptic governance violation: non-append write mode {writes}")
    return {"imports": sorted(imported), "open_modes": sorted(set(modes)),
            "append_only": True, "can_modify_audited_artifacts": False}


# ----------------------------------------------------------------------- audit
DEFAULT_ADVERSARIAL_LEDGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "adversarial_ledger.jsonl")


def _stratum(c: Claim) -> str:
    return f"{c.ledger}|{c.verdict}"


def select_targets(scored, lottery: float, seed: int, top_k: int):
    """Top-suspicion set + a SEEDED STRATIFIED lottery over everything else.

    The lottery is the residue handler from a_skeptic Q4: the disposition we
    cannot compile is 'something smells off with no articulable feature', so a
    blind sample of low-suspicion rows is the only thing that can ever surface
    a feature the allocator does not have. Deterministic under `seed`.
    """
    ranked = sorted(scored, key=lambda s: (-s["suspicion"], s["claim"].id()))
    top = ranked[:max(0, top_k)]
    top_ids = {s["claim"].id() for s in top}
    rest = [s for s in ranked if s["claim"].id() not in top_ids]
    strata: dict = {}
    for s in rest:
        strata.setdefault(_stratum(s["claim"]), []).append(s)
    rng = random.Random(seed)
    picked = []
    for name in sorted(strata):
        pool = sorted(strata[name], key=lambda s: s["claim"].id())
        n = max(1, int(round(len(pool) * lottery))) if pool and lottery > 0 else 0
        picked.extend(rng.sample(pool, min(n, len(pool))))
    picked.sort(key=lambda s: s["claim"].id())
    return top, picked


def audit(ledger_paths, lottery: float = 0.05, seed: int = 0, *,
          top_k: int = 8, out_path: str | None = None,
          write: bool = True, betas: dict | None = None,
          diagnosis_report_path: str | None = None,
          oracles: dict | None = None) -> dict:
    """Score every row, select the top-suspicion set plus a seeded stratified
    lottery, run every applicable move, APPEND findings to the adversarial
    ledger. Inputs are opened read-only and are byte-identical afterwards.

    v2 optional inputs, all OPT-IN so the default call is byte-identical to v1:
      betas                 fitted vector from `load_fitted_betas()`
      diagnosis_report_path per-stage lie rates instead of the uniform default
      oracles               adapters for the `oracle_audit` batteries
    """
    governance = assert_read_only_governance()
    claims = load_claims(ledger_paths)
    priors = build_priors(claims, diagnosis_report_path)

    scored = []
    for c in claims:
        s, breakdown = suspicion_score(c, priors, betas)
        scored.append({"claim": c, "suspicion": s, "breakdown": breakdown})

    top, lottery_picks = select_targets(scored, lottery, seed, top_k)

    findings: list[dict] = []
    for name in sorted(MOVES):
        findings.extend(MOVES[name](claims))
    oracle_findings = oracle_audit(oracles) if oracles else []
    findings.extend(oracle_findings)

    selected_ids = ([s["claim"].id() for s in top]
                    + [s["claim"].id() for s in lottery_picks])
    landed = [f for f in findings if f["verdict"] == INVALIDATED]

    report = {
        "schema": "detreason.skeptic.audit/v1",
        "seed": seed, "lottery": lottery, "top_k": top_k,
        "ledgers": [os.path.basename(p) for p in ledger_paths],
        "n_claims": len(claims),
        "priors": {"fix_rate_overall": round(priors.fix_rate_overall, 4),
                   "fix_rate_by_kind": {k: round(v, 4) for k, v in
                                        priors.fix_rate_by_kind.items()},
                   "lie_rate": priors.lie_rate_source,
                   "lie_rate_by_stage": dict(priors.lie_rate_by_stage),
                   "lie_rate_meta": priors.lie_rate_meta},
        "calibration": ("v2-fitted" if betas else "v1-heuristic-preCalibration"),
        "governance": governance,
        "ranking": [{"id": s["claim"].id(), "key": s["claim"].key,
                     "fn": s["claim"].fn, "verdict": s["claim"].verdict,
                     "fixed": s["claim"].fixed,
                     "suspicion": s["suspicion"],
                     "features": s["breakdown"]["features"]}
                    for s in sorted(scored,
                                    key=lambda s: (-s["suspicion"],
                                                   s["claim"].id()))],
        "selected": {"top_suspicion": [s["claim"].id() for s in top],
                     "lottery": [s["claim"].id() for s in lottery_picks],
                     "all": selected_ids},
        "findings": findings,
        "summary": {
            "n_findings": len(findings),
            "n_invalidated": len(landed),
            "n_survived": sum(1 for f in findings if f["verdict"] == SURVIVED),
            "n_unsupported": sum(1 for f in findings
                                 if f["verdict"] == UNSUPPORTED),
            "by_move": {m: {
                "INVALIDATED": sum(1 for f in findings
                                   if f["move"] == m and f["verdict"] == INVALIDATED),
                "SURVIVED": sum(1 for f in findings
                                if f["move"] == m and f["verdict"] == SURVIVED),
                "UNSUPPORTED": sum(1 for f in findings
                                   if f["move"] == m and f["verdict"] == UNSUPPORTED),
            } for m in (sorted(MOVES) + (["oracle_audit"] if oracle_findings
                                         else []))},
        },
    }

    target = out_path or DEFAULT_ADVERSARIAL_LEDGER
    if write:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:   # APPEND ONLY
            for f in findings:
                rec = dict(f)
                rec["_audit"] = {"seed": seed, "lottery": lottery,
                                 "ledgers": report["ledgers"]}
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.write(json.dumps({"move": "<audit_report>", "target_key": "-",
                                 "verdict": "SURVIVED",
                                 "evidence": {k: report[k] for k in
                                              ("schema", "seed", "lottery",
                                               "ledgers", "n_claims", "priors",
                                               "governance", "selected",
                                               "summary")}},
                                sort_keys=True) + "\n")
    report["adversarial_ledger"] = target
    return report


if __name__ == "__main__":  # pragma: no cover - manual driver
    import sys
    rep = audit(sys.argv[1:] or [], lottery=0.05, seed=1337)
    print(json.dumps(rep["summary"], indent=2))
    for row in rep["ranking"][:5]:
        print(row["suspicion"], row["id"], row["verdict"], row["fixed"])
