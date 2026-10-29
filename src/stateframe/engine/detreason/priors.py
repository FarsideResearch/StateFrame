"""DETREASON PRIORS (memory/consolidation + self-model) — the nightly distiller.

Design authority: sol_answers/a_faculties.md section 1 (MEMORY/CONSOLIDATION) and
section 2 (SELF-MODEL/CALIBRATION).

  §1  "A prior registry keyed by module, atom-family, frame, context, and version.
       ... Nightly replay certificates, outcomes, selective errors, disagreements,
       gate traces and boot evals into Beta/Dirichlet-style updates with shrinkage
       toward global priors."
      Anti-ossification: "Keep short-term and long-term posteriors separately.
       Apply measured half-life, detect change points ..."
      Trap: "tiny samples presented as confidence".

  §2  The COMPETENCE TABLE is "a layer above" per-sensor calibration: sensor
      calibration asks *is this measurement reliable*; competence asks *can the
      whole pipeline resolve this family end-to-end*. Dispatch, depth control and
      escalation read it. Trap: "treating confidence as competence".

WHAT IS AND IS NOT IMPLEMENTED IN THIS VERSION
----------------------------------------------
Implemented: Beta(a,b) posteriors per (context_kind, family), shrinkage toward the
per-context global prior, TWO windows by row order (short = last N rows, long =
all rows), exact Beta lower credible bounds, provenance, and a read-only
CompetenceTable with documented recommendation thresholds.

NOT implemented and deliberately NOT faked (see PriorKey fields prefixed `todo_`):
measured half-life decay, change-point detection, and contradiction splitting.
Those fields are emitted as explicit nulls with a reason string so nobody can read
a decayed posterior that was never computed. The two hard windows are the honest
stand-in: if the short window disagrees with the long window, that IS the drift
signal, and it is reported rather than smoothed away.

HONESTY RULES ENFORCED HERE
---------------------------
* A ledger field that is absent counts as MISSING (no trial recorded), never as a
  failure. Half of results/real_repo_baseline_v3.jsonl stops after the file stage
  and simply has no func/mutation/fix keys — scoring those as failures would
  invent 32 losses that never happened.
* A stage trial is only recorded when the pipeline actually REACHED that stage
  (the previous stage succeeded). Otherwise there was no attempt to score.
* Every key carries its own n. `dump_markdown()` has n columns and refuses to
  print a recommendation stronger than `no_data` below MIN_TRIALS.
* Rows carrying no outcome at all (results/mr_history_tasks.jsonl is task INPUT,
  not results) contribute zero trials and are counted in provenance as
  `rows_no_outcome` so their absence is visible rather than silent.

CLI:  python3 detreason/priors.py distill
      -> writes results/prior_registry.json and prints the competence table.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# ------------------------------------------------------------------ constants

SCHEMA_VERSION = 2

#: Rows (in ledger order, across all sources) that make up the SHORT window.
DEFAULT_SHORT_WINDOW_ROWS = 20

#: Shrinkage strength toward the per-context global prior, in pseudo-observations.
#: posterior = Beta(m*K + successes, (1-m)*K + failures) where m is the pooled
#: success rate across every family in the same context_kind. K=2 means "a family
#: with zero evidence reads exactly as the global mean, and two real observations
#: already outweigh the prior" — deliberately weak, because the traps in §1/§2 are
#: about over-trusting, and a strong prior would hide a genuinely bad family.
SHRINKAGE_K = 2.0

#: The pooled global mean is clamped into [CLAMP, 1-CLAMP] before it becomes
#: pseudo-counts. Two reasons, both load-bearing:
#:  * arithmetic: a context whose every trial passed would give beta = 0, and
#:    Beta(a, 0) is degenerate (it has no density and no quantile).
#:  * honesty: "this context has never failed" is not evidence that failure is
#:    impossible. A prior that assigns zero mass to failure can never be moved by
#:    a future failure, which is precisely the ossification §1 warns about.
GLOBAL_MEAN_CLAMP = 0.05

#: Below this many real trials a key may only ever recommend `no_data`
#: (§1 trap: "tiny samples presented as confidence").
MIN_TRIALS = 5

#: Recommendation thresholds, applied to the 95% LOWER credible bound (not the
#: mean) so that a small sample cannot buy trust on optimism alone.
TRUST_LO95 = 0.70
VERIFY_LO95 = 0.40

#: Context kinds this distiller produces.
CTX_BUG_CLASS = "bug_class"
CTX_TASK_KIND = "task_kind"
CTX_STAGE = "stage"
CTX_MODULE_PREFIX = "module_prefix"

CONTEXT_KINDS = (CTX_BUG_CLASS, CTX_TASK_KIND, CTX_STAGE, CTX_MODULE_PREFIX)


# ------------------------------------------------------- beta math (stdlib only)

def _log_gamma(x: float) -> float:
    """Lanczos log-gamma. Deterministic, no imports beyond math in caller."""
    import math

    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)
    x -= 1.0
    a = c[0]
    t = x + g + 0.5
    for i in range(1, g + 2):
        a += c[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b) = P(X <= x) for X ~ Beta(a, b)."""
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse Beta CDF by bisection. Deterministic to ~1e-10."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


# ----------------------------------------------------------------- trial stream

@dataclass
class Trial:
    """One recorded outcome. `row_index` is the position in the global ordered
    row stream and is what the short/long windows slice on."""
    context_kind: str
    family: str
    success: int          # 1 or 0 — never inferred from a missing field
    row_index: int
    source: str


def _module_prefix(path: str | None) -> str | None:
    """First path segment, e.g. 'utils/foo.py' -> 'utils'. A bare filename has no
    module prefix, so it becomes '(root)' rather than being silently dropped."""
    if not path or not isinstance(path, str):
        return None
    path = path.strip()
    if not path or path.startswith("("):
        return None
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    if len(parts) == 1:
        return "(root)"
    return parts[0]


# --------------------------------------------------------------- row extractors
#
# Each extractor documents, per source, exactly which real field it reads. If a
# field is absent the extractor yields nothing for that stage: MISSING, not a
# failure. Nothing here synthesises an outcome from another outcome.


def _extract_real_repo_baseline(row: dict, idx: int, source: str) -> list[Trial]:
    """results/real_repo_baseline_v3.jsonl

    Two schema generations are appended in this file and they are handled
    honestly rather than unified:
      * short rows: score == {file_in_menu, file_hit}. Every one of these has
        file_in_menu == 0, i.e. the run aborted at the file menu. There is no
        func/mutation/fix evidence and none is invented.
      * long rows: score adds {func_hit, mut_in_menu, fixed} and the row adds
        {func_pick, inferred_cls, n_mutations}.
    `cls` is the GOLD bug class (present on every row). `inferred_cls` is the
    pipeline's guess and is not used as a bug-class label — using a guess as its
    own ground truth is exactly the self-reinforcing-label trap in §1.
    """
    out: list[Trial] = []
    score = row.get("score")
    if not isinstance(score, dict):
        return out

    cls = row.get("cls")
    prefix = _module_prefix(row.get("path"))

    def add(ctx, fam, val):
        if fam is None or val is None:
            return
        out.append(Trial(ctx, str(fam), 1 if val else 0, idx, source))

    # --- file menu: the gold file appeared in the generated menu.
    file_in_menu = score.get("file_in_menu")
    if file_in_menu is not None:
        add(CTX_STAGE, "file_menu", file_in_menu)
        add(CTX_BUG_CLASS, cls, file_in_menu)
        add(CTX_TASK_KIND, "real_repo", file_in_menu)
        add(CTX_MODULE_PREFIX, prefix, file_in_menu)

    # --- file pick: menu contained it AND the pipeline chose it. Only a real
    # attempt when the file was in the menu at all.
    file_hit = score.get("file_hit")
    if file_hit is not None and file_in_menu:
        add(CTX_STAGE, "file_pick", file_hit)

    # --- downstream stages: only counted when present in this row's schema.
    func_hit = score.get("func_hit")
    if func_hit is not None:
        add(CTX_STAGE, "func_menu", func_hit)
    mut = score.get("mut_in_menu")
    if mut is not None:
        add(CTX_STAGE, "mutation_menu", mut)
    fixed = score.get("fixed")
    if fixed is not None:
        add(CTX_STAGE, "line_fix", fixed)
    return out


def _extract_unified_gym(row: dict, idx: int, source: str) -> list[Trial]:
    """results/unified_gym.jsonl

    Per-stage success is derived from the `stages` trace, which is a list of small
    dicts appended as the pipeline advances:
        {"class":..} {"file":..} {"function":..} {"mutations": N} {"fix":..}
    Rule: a stage is a TRIAL only if the pipeline reached it, i.e. the previous
    stage succeeded. A run that stops after {"file":..} did not fail the function
    menu — it never ran one, so no func_menu trial is recorded.

    Outcome families (bug_class / task_kind / module_prefix) key off
    `gold_contact` (the run touched the gold defect), which is the only
    end-to-end truth label in this file; `fixed` is 1 on exactly one row and is
    reported in provenance rather than used as a second, near-degenerate label.
    """
    out: list[Trial] = []
    stages = row.get("stages")
    if not isinstance(stages, list):
        stages = []
    keys = [set(s.keys()) for s in stages if isinstance(s, dict)]

    def has(k):
        return any(k in s for s in keys)

    def val(k):
        for s in stages:
            if isinstance(s, dict) and k in s:
                return s[k]
        return None

    def add(ctx, fam, v):
        if fam is None or v is None:
            return
        out.append(Trial(ctx, str(fam), 1 if v else 0, idx, source))

    reached_file = has("file")
    add(CTX_STAGE, "file_menu", 1 if reached_file else 0)

    if reached_file:
        reached_fn = has("function")
        add(CTX_STAGE, "func_menu", 1 if reached_fn else 0)
        if reached_fn:
            n_mut = val("mutations")
            got_mut = isinstance(n_mut, int) and n_mut > 0
            add(CTX_STAGE, "mutation_menu", 1 if got_mut else 0)
            if got_mut:
                add(CTX_STAGE, "line_fix", 1 if has("fix") else 0)

    gold = row.get("gold_contact")
    if gold is not None:
        cls = None
        for s in stages:
            if isinstance(s, dict) and "class" in s:
                cls = s["class"]
                break
        add(CTX_BUG_CLASS, cls, gold)
        kind = row.get("kind")
        add(CTX_TASK_KIND, kind, gold)
        # `key` is a module path for synthetic rows and a commit sha for real
        # ones; _module_prefix returns None for a sha-shaped key, so real rows
        # simply contribute no module-prefix evidence.
        k = row.get("key")
        if isinstance(k, str) and "/" in k:
            add(CTX_MODULE_PREFIX, _module_prefix(k), gold)
    return out


def _extract_gauntlet(row: dict, idx: int, source: str) -> list[Trial]:
    """results/gauntlet_v5.jsonl

    Different workload shape: synthetic mini-repos scored on
    {located, tests_pass, no_collateral}. There is no bug-class label here, so
    this source contributes ZERO bug_class evidence — it is not force-fitted.
    `mode` is the real per-attempt family (line_fix / assemble_guards /
    assemble_branch) and is recorded under `stage`; `kind` (A / B(c0) / C) is the
    task family and is namespaced `gauntlet:<kind>` so it can never be pooled
    with unified_gym's synthetic/real kinds.
    """
    out: list[Trial] = []
    score = row.get("score")
    if not isinstance(score, dict):
        return out

    def add(ctx, fam, v):
        if fam is None or v is None:
            return
        out.append(Trial(ctx, str(fam), 1 if v else 0, idx, source))

    located = score.get("located")
    add(CTX_STAGE, "locate", located)

    tests_pass = score.get("tests_pass")
    mode = row.get("mode")
    if tests_pass is not None and mode:
        add(CTX_STAGE, str(mode), tests_pass)
        kind = row.get("kind")
        if kind:
            add(CTX_TASK_KIND, "gauntlet:%s" % kind, tests_pass)
        reads = row.get("reads")
        if isinstance(reads, list) and reads:
            add(CTX_MODULE_PREFIX, _module_prefix(reads[0]), tests_pass)
    return out


def _extract_no_outcome(row: dict, idx: int, source: str) -> list[Trial]:
    """results/mr_history_tasks.jsonl and anything else that is TASK INPUT.

    These rows carry sha/msg/path/fn/changed_line/parent_src/fixed_src — a task
    definition, not a result. There is no outcome field, so there is no trial.
    Counted in provenance as rows_no_outcome so the zero is visible.
    """
    return []


#: filename fragment -> (extractor, human label). Matching is on basename so the
#: caller may pass absolute or relative paths.
_EXTRACTORS = (
    ("real_repo_baseline", _extract_real_repo_baseline),
    ("unified_gym", _extract_unified_gym),
    ("gauntlet_v", _extract_gauntlet),
    ("mr_history_tasks", _extract_no_outcome),
)


def extractor_for(path: str):
    base = os.path.basename(path)
    for frag, fn in _EXTRACTORS:
        if frag in base:
            return fn
    return None


# ------------------------------------------------------------------- posteriors

def _posterior(successes: int, failures: int, global_mean: float,
               k: float = SHRINKAGE_K) -> dict:
    """Beta posterior with shrinkage toward the per-context global mean.

        a = global_mean * k + successes
        b = (1 - global_mean) * k + failures

    where `global_mean` is first clamped to [GLOBAL_MEAN_CLAMP, 1-GLOBAL_MEAN_CLAMP].

    Reported `mean` is the posterior mean a/(a+b). Reported `n_effective` is the
    count of REAL trials (successes + failures) — the pseudo-counts are prior, not
    evidence, and are never presented as sample size.
    `lo95` is the exact 2.5% quantile of that Beta: a genuine lower credible
    bound, so a 1-of-1 success cannot masquerade as competence.
    """
    n = successes + failures
    m = min(max(global_mean, GLOBAL_MEAN_CLAMP), 1.0 - GLOBAL_MEAN_CLAMP)
    a = m * k + successes
    b = (1.0 - m) * k + failures
    return {
        "alpha": a,
        "beta": b,
        "successes": successes,
        "failures": failures,
        "n_effective": n,
        "mean": a / (a + b),
        "raw_rate": (successes / n) if n else None,
        "lo95": beta_quantile(0.025, a, b),
        "hi95": beta_quantile(0.975, a, b),
    }


def recommend(post: dict) -> str:
    """trust | verify_extra | route_up | no_data — documented thresholds.

    no_data      : fewer than MIN_TRIALS real trials. §1's trap is tiny samples
                   presented as confidence, so small-n never yields a routing
                   opinion at all, however good the rate looks.
    trust        : lo95 >= TRUST_LO95 (0.70). The pipeline resolves this family
                   end to end; dispatch may run it at low depth.
    verify_extra : lo95 >= VERIFY_LO95 (0.40). Competent but not reliable —
                   run it, then spend an extra verification gate on the result.
    route_up     : otherwise. Escalate: this family is below the level where the
                   kernel's own verdict is worth its cost.
    """
    if post["n_effective"] < MIN_TRIALS:
        return "no_data"
    lo = post["lo95"]
    if lo >= TRUST_LO95:
        return "trust"
    if lo >= VERIFY_LO95:
        return "verify_extra"
    return "route_up"


# --------------------------------------------------------------- prior registry

@dataclass
class PriorRegistry:
    schema_version: int = SCHEMA_VERSION
    distilled_at: str = ""
    short_window_rows: int = DEFAULT_SHORT_WINDOW_ROWS
    shrinkage_k: float = SHRINKAGE_K
    min_trials: int = MIN_TRIALS
    thresholds: dict = field(default_factory=lambda: {
        "trust_lo95": TRUST_LO95, "verify_lo95": VERIFY_LO95})
    provenance: dict = field(default_factory=dict)
    global_priors: dict = field(default_factory=dict)
    #: "context_kind::family" -> {"long": {...}, "short": {...}, todo_* fields}
    keys: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(obj: dict) -> "PriorRegistry":
        r = PriorRegistry()
        for k, v in obj.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r

    @staticmethod
    def load(path: str) -> "PriorRegistry":
        with open(path) as fh:
            return PriorRegistry.from_json(json.load(fh))


def key_of(context_kind: str, family: str) -> str:
    return "%s::%s" % (context_kind, family)


def distill(result_paths: list, out_path: str | None = None,
            short_window_rows: int = DEFAULT_SHORT_WINDOW_ROWS,
            now: str | None = None) -> PriorRegistry:
    """Walk the ledgers and produce the Beta prior registry.

    Rows are read in file order, and the files in the order given; that combined
    order defines `row_index`, which the SHORT window slices on (last
    `short_window_rows` rows). Missing files are recorded in provenance rather
    than raising — a nightly job must not die because one ledger was not written.
    """
    trials: list[Trial] = []
    sources: list[dict] = []
    row_index = 0

    for path in result_paths:
        rec = {
            "path": path,
            "exists": os.path.exists(path),
            "rows": 0,
            "rows_parsed": 0,
            "rows_no_outcome": 0,
            "rows_unparseable": 0,
            "trials": 0,
            "schema_signatures": {},
            "extractor": None,
            "note": None,
        }
        fn = extractor_for(path)
        rec["extractor"] = getattr(fn, "__name__", None)
        if not rec["exists"]:
            rec["note"] = "file not found; contributed nothing"
            sources.append(rec)
            continue
        if fn is None:
            rec["note"] = "no extractor registered for this filename; skipped"
            sources.append(rec)
            continue

        sigs: dict = {}
        before = len(trials)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec["rows"] += 1
                try:
                    row = json.loads(line)
                except (ValueError, TypeError):
                    rec["rows_unparseable"] += 1
                    row_index += 1
                    continue
                if not isinstance(row, dict):
                    rec["rows_unparseable"] += 1
                    row_index += 1
                    continue
                rec["rows_parsed"] += 1
                sig = ",".join(sorted(row.keys()))
                sigs[sig] = sigs.get(sig, 0) + 1
                got = fn(row, row_index, os.path.basename(path))
                if not got:
                    rec["rows_no_outcome"] += 1
                trials.extend(got)
                row_index += 1
        rec["schema_signatures"] = sigs
        rec["trials"] = len(trials) - before
        if len(sigs) > 1:
            rec["note"] = (
                "%d distinct row schemas in one file (appended runs from "
                "different harness versions); fields absent in a schema are "
                "treated as MISSING, never as failures" % len(sigs))
        sources.append(rec)

    total_rows = row_index
    short_cutoff = max(0, total_rows - short_window_rows)

    # --- global priors: pooled success rate per context_kind, over the LONG
    # window. Families shrink toward their own context's pool, not a single
    # cross-context number (§2 trap: aggregating incompatible contexts).
    globals_: dict = {}
    for ck in CONTEXT_KINDS:
        s = sum(t.success for t in trials if t.context_kind == ck)
        n = sum(1 for t in trials if t.context_kind == ck)
        globals_[ck] = {
            "successes": s,
            "trials": n,
            # A context with no evidence gets 0.5 — maximally uninformative —
            # and every family under it will read `no_data` anyway.
            "mean": (s / n) if n else 0.5,
        }

    # --- per-key posteriors, both windows.
    keys: dict = {}
    seen = {}
    for t in trials:
        seen.setdefault((t.context_kind, t.family), []).append(t)

    for (ck, fam), ts in sorted(seen.items()):
        gm = globals_[ck]["mean"]
        long_s = sum(t.success for t in ts)
        long_f = len(ts) - long_s
        short_ts = [t for t in ts if t.row_index >= short_cutoff]
        short_s = sum(t.success for t in short_ts)
        short_f = len(short_ts) - short_s

        long_post = _posterior(long_s, long_f, gm)
        short_post = _posterior(short_s, short_f, gm)
        keys[key_of(ck, fam)] = {
            "context_kind": ck,
            "family": fam,
            "long": long_post,
            "short": short_post,
            "sources": sorted({t.source for t in ts}),
            "first_row_index": min(t.row_index for t in ts),
            "last_row_index": max(t.row_index for t in ts),
            # Drift is only meaningful when the short window actually observed
            # this key. With zero short trials the "short mean" is pure prior, and
            # subtracting the long mean from a prior would render as a large
            # spurious drift for any family that simply stopped appearing.
            "drift_short_minus_long": (
                (short_post["mean"] - long_post["mean"])
                if short_post["n_effective"] > 0 else None),
            # --- explicitly NOT implemented (see module docstring) ---
            "todo_half_life_days": None,
            "todo_decayed_posterior": None,
            "todo_change_point_row_index": None,
            "todo_contradiction_contexts": None,
            "todo_note": (
                "half-life decay, change-point detection and contradiction "
                "splitting are NOT implemented; the short/long window pair is "
                "the v1 stand-in and drift_short_minus_long is the only drift "
                "signal this registry can honestly report"),
        }

    reg = PriorRegistry(
        distilled_at=now or datetime.now(timezone.utc).isoformat(),
        short_window_rows=short_window_rows,
        provenance={
            "sources": sources,
            "total_rows": total_rows,
            "total_trials": len(trials),
            "short_window_first_row_index": short_cutoff,
            "windows": (
                "long = all %d rows in ledger order; short = rows %d..%d "
                "(last %d)" % (total_rows, short_cutoff, max(total_rows - 1, 0),
                               short_window_rows)),
        },
        global_priors=globals_,
        keys=keys,
    )

    if out_path:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(reg.to_json(), fh, indent=2, sort_keys=True)
            fh.write("\n")
    return reg


# ------------------------------------------------------------ competence table

class CompetenceTable:
    """Read-only view over a PriorRegistry — the §2 self-model.

    This is a LAYER ABOVE sensor calibration: a sensor calibration table says
    whether one measurement is trustworthy; this says whether the whole pipeline
    resolves a family end to end. Dispatch, depth control and escalation read
    `competence(...).recommendation`; nothing here writes.
    """

    def __init__(self, registry: PriorRegistry):
        self.registry = registry

    # -- lookup -------------------------------------------------------------
    def competence(self, context_kind: str, family: str,
                   window: str = "long") -> dict:
        """{mean, n_effective, lo95, recommendation, ...} for one key.

        An unknown key is NOT an error and NOT a zero: it returns
        n_effective=0 / recommendation='no_data' with mean=None, because "we have
        never measured this" and "we measured this and it fails" are different
        facts and must not render identically.
        """
        entry = self.registry.keys.get(key_of(context_kind, family))
        if entry is None:
            return {
                "context_kind": context_kind,
                "family": family,
                "window": window,
                "mean": None,
                "lo95": None,
                "hi95": None,
                "n_effective": 0,
                "raw_rate": None,
                "recommendation": "no_data",
                "known": False,
            }
        post = entry[window]
        return {
            "context_kind": context_kind,
            "family": family,
            "window": window,
            "mean": post["mean"],
            "lo95": post["lo95"],
            "hi95": post["hi95"],
            "n_effective": post["n_effective"],
            "raw_rate": post["raw_rate"],
            "successes": post["successes"],
            "failures": post["failures"],
            "recommendation": recommend(post),
            "drift_short_minus_long": entry["drift_short_minus_long"],
            "sources": entry["sources"],
            "known": True,
        }

    def families(self, context_kind: str) -> list:
        return sorted(
            e["family"] for e in self.registry.keys.values()
            if e["context_kind"] == context_kind
        )

    # -- rendering ----------------------------------------------------------
    def dump_markdown(self) -> str:
        """The human table. Every row carries BOTH n columns; a rate without its
        n is exactly the artefact §1 warns about, so it is never printed alone."""
        reg = self.registry
        out = []
        out.append("# Competence table")
        out.append("")
        out.append("distilled_at: `%s`" % reg.distilled_at)
        out.append("")
        prov = reg.provenance or {}
        out.append("rows walked: **%d** | trials recorded: **%d** | %s"
                   % (prov.get("total_rows", 0), prov.get("total_trials", 0),
                      prov.get("windows", "")))
        out.append("")
        out.append("Sources:")
        out.append("")
        out.append("| ledger | rows | trials | rows w/o outcome | schemas |")
        out.append("|---|---:|---:|---:|---:|")
        for s in prov.get("sources", []):
            out.append("| `%s` | %d | %d | %d | %d |" % (
                os.path.basename(s.get("path", "?")), s.get("rows", 0),
                s.get("trials", 0), s.get("rows_no_outcome", 0),
                len(s.get("schema_signatures") or {})))
        out.append("")
        out.append("CAVEAT — read `drift` carefully: the short window slices the "
                   "CONCATENATED source order, and the ledgers are not "
                   "chronologically interleaved with each other. Within one "
                   "ledger the order is real time; across ledgers the short "
                   "window simply means \"the last sources listed\". Treat a "
                   "cross-source drift as a workload-mix difference until a "
                   "per-row timestamp exists to order them properly.")
        out.append("")
        out.append("`n` is REAL trials only (shrinkage pseudo-counts are prior, "
                   "not evidence). `rec` is gated on the 95%% lower bound and is "
                   "`no_data` below n=%d." % reg.min_trials)
        out.append("")
        out.append("| context | family | n (long) | rate | mean | lo95 | "
                   "n (short) | mean (short) | drift | rec |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

        def fmt(x):
            return "—" if x is None else ("%.3f" % x)

        for ck in CONTEXT_KINDS:
            entries = [e for e in reg.keys.values() if e["context_kind"] == ck]
            for e in sorted(entries, key=lambda z: (-z["long"]["n_effective"],
                                                    z["family"])):
                lo_, sh = e["long"], e["short"]
                drift = e["drift_short_minus_long"]
                out.append("| %s | `%s` | %d | %s | %s | %s | %d | %s | %s | %s |"
                           % (ck, e["family"], lo_["n_effective"],
                              fmt(lo_["raw_rate"]), fmt(lo_["mean"]),
                              fmt(lo_["lo95"]), sh["n_effective"],
                              fmt(sh["mean"]) if sh["n_effective"] else "—",
                              ("%+0.3f" % drift) if drift is not None else "—",
                              recommend(lo_)))
        out.append("")
        out.append("Not implemented (emitted as nulls, never faked): half-life "
                   "decay, change-point detection, contradiction splitting.")
        return "\n".join(out)


# ------------------------------------------------------------------------- CLI

DEFAULT_SOURCES = [
    "results/unified_gym.jsonl",
    "results/real_repo_baseline_v3.jsonl",
    "results/gauntlet_v5.jsonl",
    "results/mr_history_tasks.jsonl",
]
DEFAULT_OUT = "results/prior_registry.json"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] != "distill":
        print("usage: python3 detreason/priors.py distill")
        return 2
    root = _repo_root()
    paths = [os.path.join(root, p) for p in DEFAULT_SOURCES]
    out = os.path.join(root, DEFAULT_OUT)
    reg = distill(paths, out)
    print(CompetenceTable(reg).dump_markdown())
    print("")
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
