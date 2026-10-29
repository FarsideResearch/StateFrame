"""WORLD MODEL v1 — predict the blast radius BEFORE the edit, then score the
prediction against what actually broke.

Committed design: `sol_answers/w8/a_world_model.md`. Three commitments from
that answer are load-bearing here and are implemented literally:

  Q1  impact(x) is a *capped-depth* reverse-dependency closure over the typed
      relation substrate (`detreason/relations.py`), NOT a bespoke graph. The
      store is built from the two silos that already exist — the seam manifest
      (`chains.seam_manifest`) and the function index
      (`func_index.build_func_index`) — through the adapters
      `relations.unified_store` already owns. This module adds NO new graph.

  Q2  Prediction is compared against REALITY, and reality is the *newly
      introduced* seam violations (before/after diff, so a pre-existing
      violation is never charged to the edit) plus, when available, failing
      tests. The list that matters is `surprises`: breakage OUTSIDE the
      predicted set. A false negative is the expensive error; an
      overprediction is merely wasteful.

  Q3  Everything here is pure AST/graph. No runtime tracing, no model call,
      no subprocess. Reflection, dynamic imports and plugin wiring are
      therefore INVISIBLE to it, which is exactly what the surprise list
      exists to measure.

THE HONESTY RULE (crucible, non-negotiable and tested):
a good calibration score may not gate anything until the ledger has enough
observations to mean something. `calibration()` always reports `n`, the floor,
and a `trusted` flag; `trusted` is False below the floor NO MATTER how good
the Brier score looks. A scorecard computed from three runs is a rumour.

Determinism: every list this module returns is sorted, every fingerprint is a
sha256 of a canonical JSON dump. Two runs over the same inputs produce
byte-identical output, because a deterministic kernel cannot consume a
decision input that reorders itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import chains
import func_index
from detreason import relations as rel

POC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(POC, "results")
DEFAULT_LEDGER = os.path.join(RESULTS, "world_model_calibration.jsonl")

#: Traversal cap. The design commits to a maximum depth; past this the
#: reachable set is everything and the prediction stops being a prediction.
#: A capped walk is recorded as such (`basis["depth_capped"]`) rather than
#: silently presented as a complete closure.
MAX_DEPTH = 3

#: Per-hop decay applied on top of the substrate's own path confidence:
#: impact at distance is a strictly weaker claim than impact next door.
DEPTH_DECAY = 0.7

#: Affected-set size thresholds for the base risk tier.
TIER_MED_AT = 3
TIER_HIGH_AT = 10

#: Minimum recorded observations before `calibration()` may be trusted to
#: gate anything. Below this the scorecard is reported and explicitly NOT
#: trusted — see THE HONESTY RULE above.
MIN_CALIBRATION_N = 20

RISK_TIERS = ("LOW", "MED", "HIGH")

#: tier -> the verification depth the attention allocator should buy. This is
#: a plain dict deliberately: the allocator consumes it via `task_state`, so
#: there is NO import edge between the two modules in either direction.
#: Documented shape of `verify_depth_hint`:
#:     {"tier": str,                 # LOW | MED | HIGH
#:      "depth": int,                # 1..3, monotone in tier
#:      "checks": tuple[str, ...],   # named verification stages to run
#:      "max_targets": int | None,   # cap on files to verify (None = all)
#:      "auto_accept_allowed": bool} # HIGH never auto-accepts
VERIFY_DEPTH = {
    "LOW": dict(depth=1,
                checks=("targeted_tests", "witness_replay"),
                max_targets=3,
                auto_accept_allowed=True),
    "MED": dict(depth=2,
                checks=("affected_tests", "seam_validation",
                        "mutation_family", "witness_replay"),
                max_targets=12,
                auto_accept_allowed=True),
    "HIGH": dict(depth=3,
                 checks=("full_chain", "seam_validation", "mutation_family",
                         "skeptic_audit", "witness_replay"),
                 max_targets=None,
                 auto_accept_allowed=False),
}


# ------------------------------------------------------------------ helpers

def _sha(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_test_node(node: str) -> bool:
    """Is this node id inside a test file?

    Node ids are either a path or ``path::qualname`` (see
    ``relations.seam_node_id``) or ``path::qualname::lineno`` (func_index), so
    the path is always the prefix up to the first ``::``.
    """
    path = node.split("::", 1)[0]
    base = os.path.basename(path)
    return (base.startswith("test_") or base.endswith("_test.py")
            or "/tests/" in path or path.startswith("tests/"))


def _is_artifact(node: str, files: dict) -> bool:
    """Is this node a real editable/testable artifact rather than a waypoint?

    Artifacts are repo files and symbol ids anchored to a repo file. A bare
    module NAME (``"app"``, ``"os"``) is neither: it is the alias bridge's
    waypoint or an external package, and reporting it as "affected" would put
    a node in the blast radius that no verification stage can act on.
    """
    if node in files:
        return True
    return "::" in node and node_file(node) in files


def node_file(node: str) -> str:
    return node.split("::", 1)[0]


def violation_key(v: dict) -> tuple:
    """Identity of a violation for before/after diffing.

    Deliberately EXCLUDES lineno: an edit that shifts a file's lines must not
    make every untouched violation in it look new. `chains._violation_key`
    makes the same choice for the same reason.
    """
    return (v.get("kind"), v.get("file"), v.get("name"))


# ------------------------------------------------------------------ records

@dataclass(frozen=True)
class Affected:
    """One predicted-affected node plus the proof it was reached."""

    target: str
    path_evidence: tuple      # tuple[dict] — the relation atoms traversed
    weight: float             # [0,1]: how strongly we predict breakage here
    depth: int

    @property
    def sort_key(self) -> tuple:
        return (-self.weight, self.depth, self.target)

    def as_dict(self) -> dict:
        return dict(target=self.target, path_evidence=list(self.path_evidence),
                    weight=round(self.weight, 6), depth=self.depth)


@dataclass(frozen=True)
class Prediction:
    """What we believe will feel this edit, and why."""

    affected: tuple           # tuple[Affected], sorted
    risk_tier: str
    basis: dict
    edit_fingerprint: str
    prediction_id: str

    @property
    def targets(self) -> tuple:
        return tuple(a.target for a in self.affected)

    def as_dict(self) -> dict:
        return dict(prediction_id=self.prediction_id,
                    edit_fingerprint=self.edit_fingerprint,
                    risk_tier=self.risk_tier,
                    affected=[a.as_dict() for a in self.affected],
                    basis=self.basis)


@dataclass(frozen=True)
class Observation:
    """What actually happened, scored against the prediction."""

    predicted_hit: dict       # target -> bool (did this predicted node break?)
    surprises: tuple          # THE list that matters: breakage we did not see
    brier_points: tuple       # tuple[float], one per scored item
    breakage: tuple           # every attributed breakage node, sorted
    new_violations: tuple     # tuple[dict], the before/after diff
    failed_tests: tuple

    @property
    def brier(self) -> float:
        pts = self.brier_points
        return sum(pts) / len(pts) if pts else 0.0

    def as_dict(self) -> dict:
        return dict(predicted_hit=dict(sorted(self.predicted_hit.items())),
                    surprises=list(self.surprises),
                    brier_points=[round(p, 6) for p in self.brier_points],
                    brier=round(self.brier, 6),
                    breakage=list(self.breakage),
                    new_violations=list(self.new_violations),
                    failed_tests=list(self.failed_tests))


# --------------------------------------------------------------- prediction

def _normalize_edit(edit) -> dict:
    """Accept the shapes callers actually have; normalize to one.

    Accepted:
      {"file": "a.py"}                          whole-file edit
      {"file": "a.py", "qualname": "f"}         one symbol
      {"file": "a.py", "symbols": ["f", "g"]}   several
      {"files": ["a.py", "b.py"]}               multi-file
      "a.py"                                    bare path
    Anything else raises: an edit we cannot seed from would silently produce
    an EMPTY affected set, which reads as "LOW risk, nothing to check" — the
    exact fail-plausible shape this whole module exists to avoid.
    """
    if isinstance(edit, str):
        edit = {"file": edit}
    if not isinstance(edit, dict):
        raise TypeError("edit must be a dict or a path string")
    files = []
    if edit.get("file"):
        files.append(edit["file"])
    files.extend(edit.get("files") or [])
    symbols = list(edit.get("symbols") or [])
    if edit.get("qualname"):
        symbols.append(edit["qualname"])
    if not files:
        raise ValueError("edit names no file; cannot seed an impact walk")
    return dict(files=sorted(set(files)), symbols=sorted(set(symbols)),
                kind=edit.get("kind", "edit"))


def module_alias_relations(files: dict, manifest: dict) -> list:
    """Bridge atoms joining an import's MODULE NAME to the FILE that provides it.

    Why this is needed and why it belongs here rather than in the adapters:
    `relations_from_seam_manifest` emits ``importer.py --imports--> "app"`` —
    the module NAME the source literally wrote, which is the only thing that
    file asserts. Nothing in the substrate says the name ``app`` is the file
    ``app.py``, so a reverse walk from ``app.py`` reaches its importers only by
    accident. Without this bridge every impact prediction stops at the direct
    caller and reports LOW on an edit that reaches half the tree — a
    fail-plausible answer, the worst kind.

    Resolution goes through `chains.module_to_path`, the same resolver
    `seam_violations` uses, so the graph and the gate agree on what a module
    name means. Unresolvable (external) modules emit nothing: an alias we
    cannot prove is an opaque frontier, not an edge.

    Direction: ``module_name --depends_on--> path``. `impact()` walks
    in-edges, so from ``app.py`` it steps to the alias node ``app`` and from
    there to every file that imported that name.
    """
    out, seen = [], set()
    for path in sorted(manifest.get("imports") or {}):
        binds = manifest["imports"][path] or {}
        for local in sorted(binds):
            module = (binds[local] or {}).get("module")
            if not module:
                continue
            target = chains.module_to_path(module, files)
            if not target or target == module or (module, target) in seen:
                continue
            seen.add((module, target))
            out.append(rel.Relation(module, "depends_on", target,
                                    "world_model:module_alias"))
    return sorted(out, key=lambda r: r.sort_key)


def build_store(files: dict):
    """The relation store for a repo snapshot. Both silos, one graph.

    Returns (store, manifest, index) — the manifest comes back because the
    caller almost always wants `seam_violations` from the same snapshot and
    re-parsing the tree twice is pure waste.
    """
    manifest = chains.seam_manifest(files)
    index = func_index.build_func_index(files)
    store = rel.unified_store(manifest, index)
    store.add_all(module_alias_relations(files, manifest))
    return store, manifest, index


def _seed_nodes(norm: dict, store) -> tuple:
    """Seed ids: the edited files plus every named symbol we can resolve.

    A symbol that resolves to no node is NOT dropped silently — it is
    reported in `basis["unresolved_symbols"]`, because an unresolvable seed
    means the closure is incomplete and the tier must not pretend otherwise.
    """
    seeds, unresolved = [], []
    known = set(store.nodes())
    for path in norm["files"]:
        seeds.append(path)
    for qual in norm["symbols"]:
        hits = [rel.seam_node_id(p, qual) for p in norm["files"]]
        found = [h for h in hits if h in known]
        if found:
            seeds.extend(found)
        else:
            unresolved.append(qual)
    return sorted(set(seeds)), sorted(unresolved)


def predict_impact(files: dict, edit, *, depth: int = MAX_DEPTH) -> Prediction:
    """Predict the blast radius of ``edit`` over the snapshot ``files``.

    Pure: parses, walks the relation store, returns. Nothing is written and
    nothing is executed.
    """
    norm = _normalize_edit(edit)
    store, manifest, _index = build_store(files)
    seeds, unresolved = _seed_nodes(norm, store)

    # reverse-dependency closure from every seed; best (highest weight) wins
    best: dict = {}
    frontier_capped = False
    for seed in seeds:
        for d in store.impact(seed, depth=depth):
            if d.target in seeds:
                continue          # a seed is edited, not "affected by" itself
            if not _is_artifact(d.target, files):
                # module-NAME nodes (the alias bridge, and external modules)
                # are traversal waypoints, not artifacts anyone can edit or
                # test. Walk through them; never report them as affected.
                continue
            weight = d.confidence * (DEPTH_DECAY ** (d.depth - 1))
            prior = best.get(d.target)
            if prior is None or weight > prior.weight:
                best[d.target] = Affected(
                    target=d.target,
                    path_evidence=tuple(r.as_dict() for r in d.path),
                    weight=weight, depth=d.depth)
            if d.depth == depth:
                frontier_capped = True

    affected = tuple(sorted(best.values(), key=lambda a: a.sort_key))
    covered = [a.target for a in affected if is_test_node(a.target)]
    tier, tier_reason = _risk_tier(affected, covered, unresolved,
                                   frontier_capped)

    basis = dict(
        seeds=seeds,
        unresolved_symbols=unresolved,
        depth_cap=depth,
        depth_capped=frontier_capped,
        affected_count=len(affected),
        affected_files=sorted({node_file(a.target) for a in affected}),
        test_nodes=sorted(covered),
        test_covered=bool(covered),
        store_atoms=len(store),
        parse_errors=sorted(manifest.get("parse_errors") or {}),
        thresholds=dict(med_at=TIER_MED_AT, high_at=TIER_HIGH_AT,
                        decay=DEPTH_DECAY),
        tier_reason=tier_reason,
        model_version="world_model/v1",
    )
    edit_fp = _sha(norm)
    return Prediction(affected=affected, risk_tier=tier, basis=basis,
                      edit_fingerprint=edit_fp,
                      prediction_id=_sha([edit_fp, [a.as_dict()
                                                    for a in affected], tier]))


def _risk_tier(affected, covered, unresolved, capped) -> tuple:
    """Affected-set size sets the base tier; missing evidence RAISES it.

    Coverage never LOWERS a tier. A test touching an affected symbol tells you
    the breakage would be caught, not that it will not happen — and the tier
    buys verification depth, so discounting it on the strength of tests we
    have not yet run would be circular.
    """
    n = len(affected)
    if n >= TIER_HIGH_AT:
        base, why = "HIGH", f"{n} affected nodes >= {TIER_HIGH_AT}"
    elif n >= TIER_MED_AT:
        base, why = "MED", f"{n} affected nodes >= {TIER_MED_AT}"
    else:
        base, why = "LOW", f"{n} affected nodes"
    reasons = [why]
    bump = False
    if n >= TIER_MED_AT and not covered:
        bump = True
        reasons.append("no affected symbol appears in a test file "
                       "(coverage gap)")
    if unresolved:
        bump = True
        reasons.append(f"unresolved edit symbols {unresolved}: closure is "
                       f"incomplete")
    if capped:
        reasons.append("closure truncated at depth cap")
    if bump:
        base = RISK_TIERS[min(RISK_TIERS.index(base) + 1,
                              len(RISK_TIERS) - 1)]
        reasons.append(f"raised to {base}")
    return base, "; ".join(reasons)


# -------------------------------------------------------------- observation

def _diff_violations(before: list, after: list) -> list:
    """New violations only. Multiset diff on (kind, file, name).

    A violation present before the edit is quarantined, never charged to it.
    """
    counts: dict = {}
    for v in before or []:
        counts[violation_key(v)] = counts.get(violation_key(v), 0) + 1
    new = []
    for v in after or []:
        k = violation_key(v)
        if counts.get(k, 0) > 0:
            counts[k] -= 1
            continue
        new.append(v)
    return sorted(new, key=lambda v: (v.get("kind", ""), v.get("file", ""),
                                      v.get("lineno", 0), v.get("name", "")))


def _breakage_nodes(new_violations: list, failed_tests: list) -> list:
    """Attributed breakage, as node ids in the substrate's vocabulary."""
    nodes = set()
    for v in new_violations:
        f = v.get("file")
        if not f:
            continue
        nodes.add(f)
        if v.get("name"):
            nodes.add(rel.seam_node_id(f, v["name"]))
    for t in failed_tests or []:
        if isinstance(t, dict):
            f, name = t.get("file"), t.get("name")
            if f:
                nodes.add(f)
                if name:
                    nodes.add(rel.seam_node_id(f, name))
        elif isinstance(t, str) and t:
            nodes.add(t)
            nodes.add(node_file(t))
    return sorted(nodes)


def _matches(predicted: str, broken: str) -> bool:
    """Does a predicted node cover an observed breakage node?

    Coverage is at FILE granularity in one direction only: predicting the file
    covers breakage in any of its symbols (we said "this file feels it"), and
    predicting a symbol covers breakage reported against that exact symbol or
    against its file. Predicting symbol `a.py::f` does NOT cover breakage in
    `a.py::g` — claiming credit for the wrong symbol in the right file is how
    a blast-radius score flatters itself.
    """
    if predicted == broken:
        return True
    pf, bf = node_file(predicted), node_file(broken)
    if pf != bf:
        return False
    # whole-file prediction covers anything inside that file
    return "::" not in predicted


def observe_outcome(prediction: Prediction, before_violations: list,
                    after_violations: list, test_results=None) -> Observation:
    """Score a prediction against reality.

    ``test_results`` (optional) may be a list of failing test ids/dicts, or a
    dict with a ``"failed"`` key. Absent test results mean UNOBSERVED, never
    "nothing broke": they simply contribute no breakage nodes, and the Brier
    points for predicted nodes still score against the violation evidence.
    """
    failed = []
    if isinstance(test_results, dict):
        failed = list(test_results.get("failed") or [])
    elif test_results:
        failed = list(test_results)

    new_violations = _diff_violations(before_violations, after_violations)
    breakage = _breakage_nodes(new_violations, failed)

    predicted_hit, brier_points = {}, []
    for a in prediction.affected:
        hit = any(_matches(a.target, b) for b in breakage)
        predicted_hit[a.target] = hit
        brier_points.append((a.weight - (1.0 if hit else 0.0)) ** 2)

    # a surprise is breakage covered by NO predicted node and by no seed
    # (the seeds are the edit itself — breaking what you edited is not a
    # surprise, it is the point).
    covered_by = list(prediction.targets) + list(prediction.basis.get("seeds")
                                                 or [])
    raw_surprises = {b for b in breakage
                     if not any(_matches(c, b) for c in covered_by)}
    # one violation yields BOTH a file node and a symbol node; if the file is
    # itself a surprise the symbol adds no information and would charge the
    # same miss twice. Collapse to the file.
    surprises = sorted(s for s in raw_surprises
                       if "::" not in s or node_file(s) not in raw_surprises)
    # every surprise is a full-confidence miss: we predicted 0, reality is 1.
    brier_points.extend([1.0] * len(surprises))

    return Observation(predicted_hit=predicted_hit,
                       surprises=tuple(surprises),
                       brier_points=tuple(brier_points),
                       breakage=tuple(breakage),
                       new_violations=tuple(new_violations),
                       failed_tests=tuple(sorted(str(f) for f in failed)))


# ----------------------------------------------------------------- ledger

class CalibrationLedger:
    """Append-only JSONL of (prediction, observation) pairs.

    Append-only is enforced in code, not by convention: the only file mode
    this class ever opens for writing is ``"a"``, and there is no delete, no
    rewrite, and no truncate verb. A calibration history you can edit is a
    calibration history that says whatever you need it to say.
    """

    def __init__(self, path: str = DEFAULT_LEDGER,
                 min_n: int = MIN_CALIBRATION_N):
        self.path = path
        self.min_n = min_n

    # -- write ------------------------------------------------------------
    def record(self, prediction: Prediction, observation: Observation,
               meta: dict = None) -> dict:
        row = dict(prediction=prediction.as_dict(),
                   observation=observation.as_dict(),
                   meta=dict(meta or {}))
        row["row_id"] = _sha(row)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True,
                                separators=(",", ":")) + "\n")
        return row

    # -- read -------------------------------------------------------------
    def rows(self) -> list:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # a torn final line from a killed process is data loss we
                    # REPORT rather than repair: skip it, never rewrite the
                    # file to "fix" it.
                    continue
        return out

    # -- score ------------------------------------------------------------
    def calibration(self) -> dict:
        """The honest scorecard.

        `trusted` is the whole point: below `min_n` observations the numbers
        are reported but MUST NOT gate anything. A Brier of 0.02 over four
        rows is noise wearing a lab coat.
        """
        rows = self.rows()
        n = len(rows)
        pts, surprises, breakages = [], 0, 0
        predicted, predicted_missed = 0, 0
        for r in rows:
            obs = r.get("observation") or {}
            pts.extend(obs.get("brier_points") or [])
            surprises += len(obs.get("surprises") or [])
            breakages += len(obs.get("breakage") or [])
            hits = obs.get("predicted_hit") or {}
            predicted += len(hits)
            predicted_missed += sum(1 for v in hits.values() if not v)
        return dict(
            n=n,
            n_floor=self.min_n,
            trusted=n >= self.min_n,
            brier_mean=(sum(pts) / len(pts)) if pts else None,
            brier_points=len(pts),
            surprise_rate=(surprises / breakages) if breakages else 0.0,
            surprises=surprises,
            breakage_observed=breakages,
            overprediction_rate=((predicted_missed / predicted)
                                 if predicted else 0.0),
            predicted_items=predicted,
        )

    def may_gate(self) -> bool:
        """May this ledger's score influence a gate at all? n-floor only."""
        return self.calibration()["trusted"]


# ------------------------------------------------------------- entry point

def impact_check(files: dict, edit, *, depth: int = MAX_DEPTH) -> dict:
    """THE entry point. Pure, deterministic, no I/O.

    Returns::

        {"prediction": <Prediction.as_dict()>,
         "verify_depth_hint": {"tier": str, "depth": int,
                               "checks": [str, ...],
                               "max_targets": int | None,
                               "auto_accept_allowed": bool,
                               "affected_files": [str, ...]}}

    `verify_depth_hint` is a plain dict, consumed by the attention allocator
    through `task_state`. There is deliberately NO import edge between that
    module and this one in either direction: the hint is data, and a data
    contract that two modules can satisfy independently is the only kind that
    survives one of them being rewritten.
    """
    pred = predict_impact(files, edit, depth=depth)
    hint = dict(VERIFY_DEPTH[pred.risk_tier])
    hint["checks"] = list(hint["checks"])
    hint["tier"] = pred.risk_tier
    hint["affected_files"] = pred.basis["affected_files"]
    return dict(prediction=pred.as_dict(), verify_depth_hint=hint)
