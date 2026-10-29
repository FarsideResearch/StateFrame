"""Diagnosis engine — automated failure attribution over pipeline ledger rows.

Design authority: ``sol_answers/a_faculties.md`` §3 (diagnosis engine) and the ~20
manual autopsies recorded in ``FINDINGS.md`` E27-E29.

The contract, restated from a_faculties §3 and honoured literally here:

  * Replay the recorded trace against the adjudicated outcome.
  * Assign the EARLIEST causal failure, plus contributing failures.
  * Require real evidence. When the recorded fields do not support an
    attribution, return ``insufficient=True`` and NO label. Zero fabricated
    attribution is the hard invariant of this module: every branch below fires
    only on a field that is actually present in the row, and the fallback is
    silence rather than a guess.

This is the v1 of the 9-branch tree, scoped to the coding-pipeline rows our
ledgers actually contain (locate -> function -> enumerate -> fix -> gate).
Branches 5-9 of the a_faculties tree (planning, adherence, communication,
constitution, intent) have no representation in these schemas and are therefore
deliberately absent rather than guessed at.

Supported ledger schemas (auto-detected, never assumed):

  ``results/unified_gym.jsonl``
      {kind, key (gold file), fn (gold function), verdict, margins[],
       stages[], fixed, gold_contact, wall_s}
  ``results/real_repo_baseline_v3.jsonl``
      {path (gold file), fn (gold function), cls, score{file_in_menu, file_hit,
       func_hit, mut_in_menu, fixed}, rel_atoms[], file_pick, func_pick,
       inferred_cls, n_mutations}
  ``results/gauntlet_v5.jsonl``
      {level, reads[], turns, atoms[], attempts[{n, mode, raw, gate}], mode,
       func, kind, score{located, tests_pass, no_collateral, extras},
       total, max}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "attribute",
    "attribute_all",
    "CAUSE_METADATA",
    "FRAGILE_MARGIN_TAU",
]

# ---------------------------------------------------------------------------
# Cause registry: stage + fixability class + queue weight.
#
# The weight encodes "how cheaply and how reliably a fix here converts into
# recovered verdict-mass", learned from the E27-E29 autopsies: harness defects
# were repeatedly the cheapest and highest-yield fixes (line-number echo,
# enclosing-block walk), while a pure model mis-pick on a genuinely ambiguous
# menu is the most expensive thing to move.
# ---------------------------------------------------------------------------

CAUSE_METADATA: Dict[str, Dict[str, Any]] = {
    "locate.recall": {
        "stage": "locate",
        "fixability_class": "retrieval",
        "weight": 3,
        "action": "Improve candidate-pool recall (ranking/expansion) so the gold file reaches the menu at all.",
    },
    "locate.pick": {
        "stage": "locate",
        "fixability_class": "model_decision",
        "weight": 2,
        "action": "Gold file was reachable but not chosen: strengthen locate anchors/evidence ranking before the menu atom.",
    },
    "locate.nopick": {
        "stage": "locate",
        "fixability_class": "parser",
        "weight": 4,
        "action": "Gold file was IN the recorded menu but the model returned no parseable pick: this is a parser/pick-extraction defect, not a recall gap. Fix the pick-parsing layer before touching ranking.",
    },
    "function.pick": {
        "stage": "function-select",
        "fixability_class": "model_decision",
        "weight": 2,
        "action": "Correct file, wrong function: rank function anchors by task-mentioned symbols before presenting the menu.",
    },
    "fix.enumeration": {
        "stage": "enumerate",
        "fixability_class": "enumeration",
        "weight": 3,
        "action": "No candidate mutations were generated: extend the mutation-family enumerator for this bug class.",
    },
    "fix.harness": {
        "stage": "fix-apply",
        "fixability_class": "harness",
        "weight": 5,
        "action": "A correct model output was destroyed by the apply/parse layer (line-number echo family): fix the harness, not the model.",
    },
    "fix.pick.fragile": {
        "stage": "fix-select",
        "fixability_class": "fragility",
        "weight": 3,
        "action": "Thin decision margin / retry disagreement on the fix menu: route to fragility-margin abstention and order-flip retry.",
    },
    "fix.pick.model": {
        "stage": "fix-select",
        "fixability_class": "model_decision",
        "weight": 1,
        "action": "Gold-bearing menu, wrong pick on a wide margin: needs better discriminating evidence in the menu atom.",
    },
    "abstained": {
        "stage": "abstain",
        "fixability_class": "abstention",
        "weight": 1,
        "action": "Honest abstention with no earlier explained failure: coverage work, not a correctness defect.",
    },
    "gate.evidence_linkage": {
        "stage": "gate",
        "fixability_class": "gate",
        "weight": 2,
        "action": "Proposal self-downgraded for want of evidence linkage: strengthen the anchor->change linkage the gate checks.",
    },
    "success": {
        "stage": "none",
        "fixability_class": "none",
        "weight": 0,
        "action": "No failure to attribute.",
    },
}

# Below this top-2 logit margin a menu decision is treated as fragile rather
# than as a considered model choice. Calibrated placeholder; margins are
# currently logged as null by the runners, in which case this never fires and
# the row is attributed to the model instead (never to fragility on a guess).
FRAGILE_MARGIN_TAU = 0.15

# "5: def foo():" — a displayed line-number prefix echoed back into the patch.
_LINE_PREFIX_ECHO = re.compile(r"^\s*\d+\s*:\s", re.MULTILINE)

_ABSTAIN_VERDICTS = {"ABSTAIN", "ABSTAINED", "NO_FIX", "INSUFFICIENT_EVIDENCE"}
_DOWNGRADED_VERDICTS = {"LOW_CONFIDENCE_PROPOSAL", "UNGATED", "LOW_CONFIDENCE"}
_SUCCESS_VERDICTS = {"FIX_PROPOSED", "FIX_VERIFIED"}


# ---------------------------------------------------------------------------
# Normalisation: one shape, three ledgers. Every field is Optional and stays
# None when the source row does not record it — "unknown" must never collapse
# into "zero", because that is exactly how fabricated attribution happens.
# ---------------------------------------------------------------------------


@dataclass
class _Facts:
    schema: str
    key: str
    gold_file: Optional[str] = None
    gold_fn: Optional[str] = None
    file_in_menu: Optional[int] = None
    file_pick: Optional[str] = None
    file_hit: Optional[int] = None
    func_pick: Optional[str] = None
    func_hit: Optional[int] = None
    n_mutations: Optional[int] = None
    mut_in_menu: Optional[int] = None
    verdict: Optional[str] = None
    fixed: Optional[int] = None
    located: Optional[int] = None
    tests_pass: Optional[int] = None
    margins: List[Dict[str, Any]] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    present: List[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return getattr(self, name) is not None


def _detect_schema(row: Dict[str, Any]) -> str:
    if "stages" in row and "margins" in row:
        return "unified_gym"
    score = row.get("score")
    if isinstance(score, dict) and ("file_in_menu" in score or "mut_in_menu" in score):
        return "real_repo_baseline"
    if "attempts" in row and isinstance(score, dict) and "located" in score:
        return "gauntlet"
    if isinstance(score, dict) and "located" in score:
        return "gauntlet"
    return "unknown"


def _stage_value(stages: Sequence[Any], name: str) -> Optional[Any]:
    """Pull ``name`` out of the ordered stage list, or None if never recorded."""
    for stage in stages:
        if isinstance(stage, dict) and name in stage:
            return stage[name]
    return None


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _normalize(row: Dict[str, Any]) -> _Facts:
    schema = _detect_schema(row)
    facts = _Facts(schema=schema, key="")

    if schema == "unified_gym":
        stages = row.get("stages") or []
        facts.key = str(row.get("key") or row.get("fn") or "unified_gym_row")
        facts.gold_file = row.get("key")
        facts.gold_fn = row.get("fn")
        facts.file_pick = _stage_value(stages, "file")
        facts.func_pick = _stage_value(stages, "function")
        facts.n_mutations = _int_or_none(_stage_value(stages, "mutations"))
        facts.verdict = row.get("verdict")
        facts.fixed = _int_or_none(row.get("fixed"))
        facts.margins = list(row.get("margins") or [])
        # fixture_mint's cross-check found the menu contents ARE persisted in the
        # margins records (options[] was added with the membership-ledger fix):
        # derive file_in_menu from them instead of leaving it unknown. 12 rows
        # were mislabeled recall-vs-pick before this.
        for rec in facts.margins:
            if isinstance(rec, dict) and rec.get("menu") == "file" \
                    and isinstance(rec.get("options"), list):
                gold = str(facts.gold_file or "")
                facts.file_in_menu = int(any(gold and gold in str(o)
                                             for o in rec["options"]))
                break

    elif schema == "real_repo_baseline":
        score = row.get("score") or {}
        facts.key = str(row.get("path") or "real_repo_row")
        facts.gold_file = row.get("path")
        facts.gold_fn = row.get("fn")
        facts.file_in_menu = _int_or_none(score.get("file_in_menu"))
        facts.file_hit = _int_or_none(score.get("file_hit"))
        facts.func_hit = _int_or_none(score.get("func_hit"))
        facts.mut_in_menu = _int_or_none(score.get("mut_in_menu"))
        facts.fixed = _int_or_none(score.get("fixed"))
        facts.file_pick = row.get("file_pick")
        facts.func_pick = row.get("func_pick")
        facts.n_mutations = _int_or_none(row.get("n_mutations"))

    elif schema == "gauntlet":
        score = row.get("score") or {}
        level = row.get("level")
        facts.key = f"level_{level}" if level is not None else "gauntlet_row"
        facts.gold_fn = row.get("func")
        facts.located = _int_or_none(score.get("located"))
        facts.tests_pass = _int_or_none(score.get("tests_pass"))
        facts.attempts = list(row.get("attempts") or [])
        facts.func_pick = row.get("func")

    else:
        facts.key = str(row.get("key") or row.get("path") or "unknown_row")

    facts.present = [
        name
        for name in (
            "gold_file",
            "gold_fn",
            "file_in_menu",
            "file_pick",
            "file_hit",
            "func_pick",
            "func_hit",
            "n_mutations",
            "mut_in_menu",
            "verdict",
            "fixed",
            "located",
            "tests_pass",
        )
        if facts.has(name)
    ]
    return facts


# ---------------------------------------------------------------------------
# Evidence predicates. Each returns True / False / None, where None means the
# row does not record enough to answer — and a None NEVER satisfies a branch.
# ---------------------------------------------------------------------------


def _file_correct(f: _Facts) -> Optional[bool]:
    if f.file_hit is not None:
        return bool(f.file_hit)
    if f.gold_file is not None and f.file_pick is not None:
        return f.file_pick == f.gold_file
    return None


def _func_correct(f: _Facts) -> Optional[bool]:
    if f.func_hit is not None:
        return bool(f.func_hit)
    if f.gold_fn is not None and f.func_pick is not None:
        return f.func_pick == f.gold_fn
    return None


def _thin_margin(f: _Facts) -> Optional[Dict[str, Any]]:
    """Return the fragile menu record, or None. Null margins are not thin."""
    for record in f.margins:
        if not isinstance(record, dict):
            continue
        margin = record.get("margin")
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            if margin < FRAGILE_MARGIN_TAU:
                return record
    return None


def _retry_disagreement(f: _Facts) -> Optional[Dict[str, Any]]:
    """Distinct fix picks across retries on the same row == an unstable decision."""
    picks = [a.get("raw") for a in f.attempts if isinstance(a, dict) and a.get("raw")]
    if len(picks) >= 2 and len(set(picks)) >= 2:
        return {"n_attempts": len(picks), "distinct_picks": len(set(picks))}
    return None


def _harness_ate_the_fix(f: _Facts) -> Optional[Dict[str, Any]]:
    """The E27 line-number-echo family: a syntax failure whose raw output
    carries echoed line-number prefixes. That is the apply/parse layer eating a
    candidate fix, not a model decision."""
    for attempt in f.attempts:
        if not isinstance(attempt, dict):
            continue
        gate = attempt.get("gate") or ""
        raw = attempt.get("raw") or ""
        if not isinstance(gate, str) or not isinstance(raw, str):
            continue
        if gate.startswith("syntax_error") and _LINE_PREFIX_ECHO.search(raw):
            return {"attempt": attempt.get("n"), "gate": gate}
    return None


def _confidence(f: _Facts, required: Sequence[str], corroborating: Sequence[str] = ()) -> float:
    """Field completeness for this attribution.

    Required fields are what the branch fired on and are always present by
    construction; corroborating fields are the ones that would have made the
    call independently checkable. Confidence is the fraction of the union that
    the row actually recorded, floored at 0.5 so a fired branch is never
    reported as near-worthless, and never 1.0 unless everything corroborates.
    """
    fields = list(dict.fromkeys(list(required) + list(corroborating)))
    if not fields:
        return 0.5
    have = sum(1 for name in fields if f.has(name))
    return round(0.5 + 0.5 * (have / len(fields)), 2)


def _result(
    primary: Optional[str],
    *,
    confidence: float,
    evidence: Dict[str, Any],
    contributing: Optional[List[str]] = None,
    insufficient: bool = False,
) -> Dict[str, Any]:
    meta = CAUSE_METADATA.get(primary or "", {})
    return {
        "primary": primary,
        "contributing": contributing or [],
        "confidence": confidence,
        "evidence": evidence,
        "insufficient": insufficient,
        "stage": meta.get("stage"),
        "fixability_class": meta.get("fixability_class"),
    }


# ---------------------------------------------------------------------------
# The decision tree, in earliest-causal order.
# ---------------------------------------------------------------------------


def attribute(row: Dict[str, Any]) -> Dict[str, Any]:
    """Attribute one ledger row to its earliest causal failure stage.

    Returns ``{primary, contributing, confidence, evidence, insufficient,
    stage, fixability_class}``. ``primary`` is ``None`` exactly when
    ``insufficient`` is True — the module never invents a label.
    """
    if not isinstance(row, dict):
        return _result(
            None,
            confidence=0.0,
            evidence={"reason": "row is not a mapping"},
            insufficient=True,
        )

    f = _normalize(row)
    ev: Dict[str, Any] = {"schema": f.schema, "key": f.key, "fields_present": f.present}
    contributing: List[str] = []

    file_ok = _file_correct(f)
    func_ok = _func_correct(f)

    # ---- Branch 0: no failure to attribute -------------------------------
    if f.fixed == 1 or (f.tests_pass == 1 and f.located == 1):
        ev["outcome"] = "fixed" if f.fixed == 1 else "tests_pass"
        harness = _harness_ate_the_fix(f)
        if harness:
            # Succeeded, but only after the harness destroyed an earlier fix.
            contributing.append("fix.harness")
            ev["harness_echo"] = harness
        if file_ok is False:
            # The adjudicated outcome says fixed, the locate score says miss.
            # Surface the contradiction rather than letting the success label
            # bury it: one of the two signals is mis-scored and that is itself
            # a finding.
            contributing.append("locate.pick")
            ev["contradiction"] = (
                "adjudicated outcome is a pass but the locate score records a miss "
                "(gold_file=%r, file_pick=%r): one of these signals is mis-scored"
                % (f.gold_file, f.file_pick)
            )
        return _result(
            "success",
            confidence=_confidence(f, ["fixed"], ["tests_pass", "located", "verdict"]),
            evidence=ev,
            contributing=contributing,
        )

    # ---- Branch 1: locate recall — gold never reached the pool ------------
    if f.file_in_menu == 0:
        ev["file_in_menu"] = 0
        ev["gold_file"] = f.gold_file
        ev["file_pick"] = f.file_pick
        return _result(
            "locate.recall",
            confidence=_confidence(f, ["file_in_menu"], ["gold_file", "file_pick", "file_hit"]),
            evidence=ev,
            contributing=contributing,
        )

    if f.located == 0:
        ev["located"] = 0
        return _result(
            "locate.recall",
            confidence=_confidence(f, ["located"], ["gold_file", "file_pick"]),
            evidence=ev,
        )

    # A run whose trace records no file stage at all never got past locate —
    # but "no file stage" is not automatically "gold never reached the pool".
    # The locate-hammer autopsy found that in most such rows the gold file WAS
    # present in the recorded file-menu options (margins[].options); the model
    # just returned no parseable pick (or a pick recorded only on the margin
    # record, not the stage list). That is a parser/pick failure, not recall.
    # facts.file_in_menu is already derived from margins[].options during
    # normalization, so by this point it is either 1 (gold WAS in the menu) or
    # None (no options recorded on this row — recall-vs-pick genuinely not
    # separable, and the row keeps its prior behavior: locate.recall).
    if f.schema == "unified_gym" and f.file_pick is None:
        ev["file_stage"] = "absent from trace"
        if f.file_in_menu == 1:
            # Gold was reachable. Pull whatever pick the margin record itself
            # carries (stages lacked one, but margins is a separate ledger).
            menu_pick = None
            for rec in f.margins:
                if isinstance(rec, dict) and rec.get("menu") == "file":
                    menu_pick = rec.get("pick")
                    break
            ev["gold_file"] = f.gold_file
            ev["menu_pick"] = menu_pick
            if menu_pick is None:
                return _result(
                    "locate.nopick",
                    confidence=_confidence(f, ["file_in_menu"], ["gold_file", "verdict"]),
                    evidence=ev,
                )
            ev["file_pick"] = menu_pick
            return _result(
                "locate.pick",
                confidence=_confidence(f, ["file_in_menu"], ["gold_file", "verdict"]),
                evidence=ev,
            )
        return _result(
            "locate.recall",
            confidence=_confidence(f, ["verdict"], ["gold_file", "file_in_menu"]),
            evidence=ev,
        )

    # ---- Branch 2: locate pick — reachable, chosen wrong ------------------
    if file_ok is False:
        ev["gold_file"] = f.gold_file
        ev["file_pick"] = f.file_pick
        ev["file_in_menu"] = f.file_in_menu if f.has("file_in_menu") else "unknown"
        if not f.has("file_in_menu"):
            # Menu membership unrecorded: this is definitely a locate failure,
            # but recall-vs-pick is not separable from the trace. Say so.
            ev["note"] = (
                "menu membership not recorded; failure is in locate, "
                "recall-vs-pick not separable from this trace"
            )
        return _result(
            "locate.pick",
            confidence=_confidence(f, ["file_pick", "gold_file"], ["file_in_menu", "file_hit"]),
            evidence=ev,
        )

    # ---- Branch 3: function pick -----------------------------------------
    if func_ok is False:
        ev["gold_fn"] = f.gold_fn
        ev["func_pick"] = f.func_pick
        return _result(
            "function.pick",
            confidence=_confidence(f, ["func_pick", "gold_fn"], ["func_hit", "file_hit"]),
            evidence=ev,
        )

    # ---- Branch 4: enumeration — nothing to pick from ---------------------
    if f.mut_in_menu == 0 or f.n_mutations == 0:
        ev["n_mutations"] = f.n_mutations
        ev["mut_in_menu"] = f.mut_in_menu
        ev["verdict"] = f.verdict
        return _result(
            "fix.enumeration",
            confidence=_confidence(f, ["n_mutations"], ["mut_in_menu", "verdict", "fixed"]),
            evidence=ev,
        )

    # ---- Branch 4b: the harness ate a correct fix (E27 echo family) -------
    harness = _harness_ate_the_fix(f)
    if harness:
        ev["harness_echo"] = harness
        return _result(
            "fix.harness",
            confidence=_confidence(f, ["tests_pass"], ["located", "fixed"]),
            evidence=ev,
        )

    # ---- Branch 5: fix pick — gold was reachable, wrong candidate taken ---
    fix_stage_reached = (
        (f.n_mutations is not None and f.n_mutations > 0)
        or f.mut_in_menu == 1
        or bool(f.attempts)
    )
    failed_outcome = (
        f.fixed == 0
        or f.tests_pass == 0
        or (f.verdict in _SUCCESS_VERDICTS and f.fixed == 0)
    )
    if fix_stage_reached and failed_outcome:
        thin = _thin_margin(f)
        disagree = _retry_disagreement(f)
        if thin or disagree:
            ev["thin_margin"] = thin
            ev["retry_disagreement"] = disagree
            ev["n_mutations"] = f.n_mutations
            return _result(
                "fix.pick.fragile",
                confidence=_confidence(f, ["n_mutations"], ["mut_in_menu", "fixed", "verdict"]),
                evidence=ev,
            )
        ev["n_mutations"] = f.n_mutations
        ev["mut_in_menu"] = f.mut_in_menu
        ev["verdict"] = f.verdict
        ev["fixed"] = f.fixed
        if f.verdict in _DOWNGRADED_VERDICTS:
            contributing.append("gate.evidence_linkage")
        return _result(
            "fix.pick.model",
            confidence=_confidence(f, ["fixed"], ["mut_in_menu", "n_mutations", "verdict"]),
            evidence=ev,
            contributing=contributing,
        )

    # ---- Branch 6: honest abstention -------------------------------------
    if f.verdict in _ABSTAIN_VERDICTS:
        ev["verdict"] = f.verdict
        sub_reason = row.get("reason") or row.get("abstain_reason")
        cert = row.get("certificate")
        if not sub_reason and isinstance(cert, dict):
            sub_reason = cert.get("reason")
        if sub_reason:
            ev["sub_reason"] = sub_reason
        return _result(
            "abstained",
            confidence=_confidence(f, ["verdict"], ["n_mutations", "file_pick", "func_pick"]),
            evidence=ev,
        )

    # ---- Branch 7: gate downgrade ----------------------------------------
    if f.verdict in _DOWNGRADED_VERDICTS:
        ev["verdict"] = f.verdict
        return _result(
            "gate.evidence_linkage",
            confidence=_confidence(f, ["verdict"], ["fixed", "n_mutations"]),
            evidence=ev,
        )

    # ---- Branch 8: insufficient evidence. NEVER a guessed label. ---------
    ev["reason"] = "recorded fields do not support an attribution"
    ev["missing"] = [
        name
        for name in ("file_hit", "file_pick", "func_pick", "n_mutations", "verdict", "fixed", "located")
        if not f.has(name)
    ]
    return _result(None, confidence=0.0, evidence=ev, insufficient=True)


# ---------------------------------------------------------------------------
# Corpus-level report.
# ---------------------------------------------------------------------------


def _iter_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # An unparseable line is a no-op, never a guess (standing rule 10).
                yield {"__parse_error__": True, "__source_line__": lineno}
                continue
            if isinstance(parsed, dict):
                yield parsed
            else:
                yield {"__parse_error__": True, "__source_line__": lineno}


def attribute_all(
    paths: Sequence[str],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Attribute every row in ``paths`` and cluster by primary cause.

    Writes the report to ``output_path`` (default
    ``<repo>/results/diagnosis_report.json``) and returns it.
    """
    clusters: Dict[str, Dict[str, Any]] = {}
    rows_seen = 0
    parse_errors = 0
    per_source: Dict[str, Dict[str, int]] = {}

    for path in paths:
        source = os.path.basename(path)
        per_source.setdefault(source, {})
        if not os.path.exists(path):
            per_source[source]["__missing__"] = 1
            continue
        for row in _iter_rows(path):
            if row.get("__parse_error__"):
                parse_errors += 1
                continue
            rows_seen += 1
            verdict = attribute(row)
            label = verdict["primary"] if not verdict["insufficient"] else "insufficient_evidence"
            bucket = clusters.setdefault(
                label,
                {
                    "count": 0,
                    "confidences": [],
                    "exemplars": [],
                    "sources": {},
                    "stage": verdict.get("stage"),
                    "fixability_class": verdict.get("fixability_class"),
                },
            )
            bucket["count"] += 1
            bucket["confidences"].append(verdict["confidence"])
            bucket["sources"][source] = bucket["sources"].get(source, 0) + 1
            per_source[source][label] = per_source[source].get(label, 0) + 1
            if len(bucket["exemplars"]) < 5:
                bucket["exemplars"].append(
                    {"source": source, "key": verdict["evidence"].get("key"), "evidence": verdict["evidence"]}
                )

    for label, bucket in clusters.items():
        confidences = bucket.pop("confidences")
        bucket["mean_confidence"] = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        if label == "insufficient_evidence":
            bucket["stage"] = "instrumentation"
            bucket["fixability_class"] = "instrumentation"

    fix_queue = _build_fix_queue(clusters)

    report = {
        "schema_version": 1,
        "sources": list(paths),
        "rows_attributed": rows_seen,
        "parse_errors": parse_errors,
        "n_insufficient": clusters.get("insufficient_evidence", {}).get("count", 0),
        "clusters": clusters,
        "per_source": per_source,
        "fix_queue": fix_queue,
    }

    if output_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(repo_root, "results", "diagnosis_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    report["output_path"] = output_path
    return report


# Instrumentation debt is not a pipeline defect but it IS actionable, and it
# ranks high: an unattributable row is verdict-mass we cannot improve at all.
_INSUFFICIENT_META = {
    "stage": "instrumentation",
    "fixability_class": "instrumentation",
    "weight": 4,
    "action": "Rows could not be attributed: log the missing fields (menu membership, gold, outcome) so the next run is diagnosable.",
}


def _build_fix_queue(clusters: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for label, bucket in clusters.items():
        if label == "success":
            continue
        meta = _INSUFFICIENT_META if label == "insufficient_evidence" else CAUSE_METADATA.get(label, {})
        weight = meta.get("weight", 1)
        entries.append(
            {
                "primary": label,
                "stage": meta.get("stage"),
                "fixability_class": meta.get("fixability_class"),
                "count": bucket["count"],
                "weight": weight,
                "priority": bucket["count"] * weight,
                "mean_confidence": bucket["mean_confidence"],
                "action": meta.get("action"),
                "exemplar_keys": [e["key"] for e in bucket["exemplars"]],
            }
        )
    entries.sort(key=lambda e: (-e["priority"], -e["count"], e["primary"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def _default_ledgers() -> List[str]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = os.path.join(repo_root, "results")
    return [
        os.path.join(results, "unified_gym.jsonl"),
        os.path.join(results, "real_repo_baseline_v3.jsonl"),
        os.path.join(results, "gauntlet_v5.jsonl"),
    ]


if __name__ == "__main__":  # pragma: no cover - operator entry point
    report = attribute_all(_default_ledgers())
    print(json.dumps({k: v for k, v in report.items() if k != "clusters"}, indent=2))
