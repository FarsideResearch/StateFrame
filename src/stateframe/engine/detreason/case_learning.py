"""CASE_LEARNING — cross-run learning from autopsy verdicts (bounded influence).

Implements the committed design in ``/tmp/solq/w65/a_crossrun_learning.md``:

    run N -> autopsy verdict -> typed case -> run N+1 consultation
          -> BOUNDED adjustment -> decision ledger -> outcome -> hit rate
          -> reinforcement or retirement

The one invariant the whole module exists to protect:

    **A case may INFLUENCE a decision. It may never BE the decision.**

Every path here is additive and capped. A case can nudge a rank, soften a soft
refusal, or widen a family set. Nothing in this file can remove a candidate,
mark a gate pass, skip a validator, or veto. Those are structural properties of
the API, not comments: ``apply_rank_adjustments`` returns a permutation whose
baseline top candidate is provably still present, ``GateAdvice.disposition`` is
drawn from a frozen allow-list that contains no passing verdict, and
``FamilyAdvice`` only ever ADDs to a family set.

RELATIONSHIP TO ``detreason.case_memory``
-----------------------------------------
This module EXTENDS the case base; it does not fork it. ``case_memory``
remembers *solved* seams (a verified plan template, replayed and re-proven).
This module remembers *walls* (an attributed failure, consulted as bounded
evidence). They are complements with a shared spine:

  * ``LearnedCaseStore`` subclasses ``case_memory.CaseStore``, so both case
    kinds share ONE append-only JSONL log format, ONE fold-on-load discipline,
    and ONE out-of-band-edit canary (``cases_file_hash``).
  * The base class's ``op='case'`` / ``op='confirm'`` events still fold exactly
    as before; this subclass adds ``op='lcase'`` / ``op='lobs'`` /
    ``op='consult'`` / ``op='influence'`` / ``op='outcome'`` / ``op='reject'``
    and calls ``super()._fold`` for everything it does not own.
  * Names never enter a signature in ``case_memory``. Here they deliberately
    DO — an attractor case IS "utils/attempt_context.py keeps eating the pick",
    and a case that abstracted the path away would have nothing left to say.
    The scoping cost is paid honestly: learned cases are repo-scoped
    (``scope.repo_id``) and never match across repos.

WHAT IS REAL AND WHAT IS DECLARED
---------------------------------
Rule #223: a reader with no writer is a dead feature. Every field consumed here
names its producer, and where the producer does not exist yet it is said out
loud rather than guessed at:

  * ``probes/run_autopsy.py`` writes ``results/autopsy_<cycle>.jsonl`` with one
    ``record='summary'`` row and N ``record='task'`` rows carrying
    ``cycle_id / key / fn / kind / wall / lever / detail / evidence /
    taxonomy / row_verdict / shape_problems``. That is the miner's ONLY input
    and every field below is read from it. VERIFIED against the three real
    boards in ``results/`` (ref_s1, ref_partial, ref_partial2).
  * ``verdict_id`` — the autopsy does NOT stamp one. The design assumes it. So
    this module DERIVES it as ``f"{cycle_id}:{key}:{wall}"`` and says so in
    ``derive_verdict_id``. It is stable across re-runs of the same board, which
    is what the idempotence requirement actually needs. If the autopsy ever
    stamps a real id, read it (``_row_verdict_id`` prefers ``row['verdict_id']``
    when present) and the derivation becomes dead code rather than a lie.
  * ``tree_digest`` / ``scope_digest`` — computed HERE from the live files by
    ``tree_digest`` / ``scope_digest`` over ``snapshot.cse1_digest``. The
    autopsy boards carry no digest, so cases mined from an existing board get
    ``None`` and are therefore ``FRESHNESS_UNKNOWN``, which by policy means ZERO
    live influence (design: "digest unavailable -> do not apply influence").
    That is the honest reading of tonight's data: the boards prove the case
    TABLE, and a digest-carrying run is required before any case can move a
    ranking. Nothing here pretends otherwise.
  * ``certificate_ids`` — the gym rows in ``results/unified_gym.jsonl`` are
    keyed by ``key`` + ``run_at``; the autopsy row carries ``key`` but not the
    run's certificate id. ``provenance.cert_refs`` therefore records the
    ``(cycle_id, key)`` pair that identifies the certificate in the gym log.
    That is a REFERENCE, not an invented id.

OUT OF SCOPE (named so a gap is never mistaken for a bug)
---------------------------------------------------------
See ``OUT_OF_SCOPE``. Chiefly: no embeddings (structured features only, per the
design's retrieval section), no cross-repo transfer, no automatic policy
calibration (caps are declared constants and a case may narrow but never widen
them), and no writes into ``fixer.py`` (the three consultation points are
DOCUMENTED seams — see ``SEAMS`` — and remain unwired until a human wires them).
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:                      # pragma: no cover
    sys.path.insert(0, str(_ROOT))

from detreason import case_memory as cm             # noqa: E402
from detreason import snapshot as snap              # noqa: E402

# ================================================================== constants
SCHEMA = "case_learning/v1"

#: Default append-only log. Separate FILE from ``case_memory``'s so a board
#: mining run can never interleave with a live plan-case mint, but the SAME
#: format, so ``CaseStore.load``/``cases_file_hash`` work unchanged.
DEFAULT_PATH = _ROOT / "results" / "case_learning.jsonl"

#: The four typed cases. Keys are the ``case_type`` literal written to disk.
CASE_LOCATE_ATTRACTOR = "locate_attractor"
CASE_GATE_REFUSAL = "gate_refusal"
CASE_FAMILY_MISS = "family_miss"
CASE_SELECTION_MISS = "selection_miss"

CASE_TYPES = (CASE_LOCATE_ATTRACTOR, CASE_GATE_REFUSAL,
              CASE_FAMILY_MISS, CASE_SELECTION_MISS)

#: Decision stage each case may be consulted at. A case NEVER reaches a stage
#: other than its own; ``consult`` filters on this before anything else.
STAGE_LOCATE = "locate"
STAGE_FAMILY = "family_select"
STAGE_GATE = "gate"

CASE_STAGE = {
    CASE_LOCATE_ATTRACTOR: STAGE_LOCATE,
    CASE_SELECTION_MISS: STAGE_LOCATE,
    CASE_GATE_REFUSAL: STAGE_GATE,
    CASE_FAMILY_MISS: STAGE_FAMILY,
}

#: Autopsy wall -> candidate case type. ``locate_miss`` is AMBIGUOUS on purpose:
#: it covers both "a recurring hub ate the pick" (attractor) and "gold was on
#: the menu and lost on score" (selection). ``classify_row`` splits them on the
#: measured ``attractor_lock`` taxonomy tag, which ``run_autopsy`` only emits
#: when the same file won >= 2 picks across the board. Walls absent from this
#: map are never mined (``menu_miss`` maps to family_miss; ``instrumentation_
#: gap``, ``unattributed``, ``scoring_miss`` and ``arbitration_veto`` describe
#: the MEASUREMENT, not a repeatable code pattern, so they earn no case).
WALL_CASE_CANDIDATES = {
    "locate_miss": (CASE_LOCATE_ATTRACTOR, CASE_SELECTION_MISS),
    "gate_refusal": (CASE_GATE_REFUSAL,),
    "menu_miss": (CASE_FAMILY_MISS,),
    "selection_miss": (CASE_SELECTION_MISS,),
}

#: Walls that are explicitly NOT mineable, with the reason a reader would want.
UNMINEABLE_WALLS = {
    "instrumentation_gap": "the certificate is incomplete; the wall is our "
                           "instrumentation, not a code pattern to recall",
    "unattributed": "the certificate proved no boundary; nothing to state",
    "scoring_miss": "the scorer is wrong, not the repair; fix the predicate",
    "arbitration_veto": "veto semantics belong to arbitration, not to a "
                        "bounded ranking nudge",
}

#: Case lifecycle. ``probation`` is where every new case starts and where it
#: earns only ``probation_multiplier`` of its weight.
STATUS_PROBATION = "probation"
STATUS_ACTIVE = "active"
STATUS_STALE = "stale"
STATUS_RETIRED = "retired"

#: Freshness verdicts. ``unknown`` is a FIRST-CLASS state, not a missing value:
#: a case whose digests are unavailable gets zero live influence.
FRESHNESS_FRESH = "fresh"
FRESHNESS_AGED = "aged"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"

#: Outcome labels. ``neutral`` and ``unknown`` are NEVER counted as misses
#: (design: "do not retire solely because a case was not needed").
OUTCOME_HIT = "hit"
OUTCOME_MISS = "miss"
OUTCOME_NEUTRAL = "neutral"
OUTCOME_HARM = "harm"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_LABELS = (OUTCOME_HIT, OUTCOME_MISS, OUTCOME_NEUTRAL, OUTCOME_HARM,
                  OUTCOME_UNKNOWN)

#: Evaluation modes. ``control`` consults and logs but applies NOTHING.
MODE_TREATMENT = "treatment"
MODE_CONTROL = "control"
MODE_SHADOW = "shadow"
MODE_REPLAY = "replay"
EVALUATION_MODES = (MODE_TREATMENT, MODE_CONTROL, MODE_SHADOW, MODE_REPLAY)

#: The ONLY dispositions a gate case may propose. Note what is absent: there is
#: no "pass", no "accept", no "skip_validator". A gate case's strongest possible
#: effect is to keep a candidate alive for someone ELSE to judge.
GATE_DISPOSITIONS = ("retain_for_secondary_validation", "human_review",
                     "no_effect")

#: Validators a case may never soften. Kept as data so a future gate integration
#: can assert against it rather than re-deriving the rule.
NEVER_SKIP_VALIDATORS = ("compile", "regression", "security")

OUT_OF_SCOPE = {
    "embeddings": "retrieval is structured-feature only; the design permits "
                  "embeddings for recall but forbids them as the sole reason a "
                  "case influences a decision, and we have no encoder here",
    "cross_repo_transfer": "learned cases name real paths, so they are scoped "
                           "to one repo_id and never match another",
    "policy_calibration": "stage caps are declared constants; a case may "
                          "NARROW its own influence and may never widen a cap",
    "fixer_wiring": "the three consultation points are documented seams "
                    "(see SEAMS); this module edits no fixer code",
    "plan_templates": "solved-case plan shapes belong to case_memory; this "
                      "module never mints or replays a plan",
}


class MiningRefused(Exception):
    """Raised when an autopsy row does not earn a case. Carries the reason."""


# ===================================================================== policy
@dataclass(frozen=True)
class StageInfluencePolicy:
    """Versioned, per-stage influence bounds.

    Design rule: numeric limits live HERE, never in case text. A case describes
    direction and matching features; the policy decides how far that may move
    anything. ``max_rank_shift`` is what makes "cannot remove a candidate" a
    structural fact rather than a promise.
    """

    stage: str
    policy_version: int = 1
    max_single_case_delta: float = 0.08
    max_aggregate_delta: float = 0.12
    max_rank_shift: int = 2
    max_cases_per_decision: int = 6
    max_cases_per_cluster: int = 2
    probation_multiplier: float = 0.25
    #: Gate-only. Soft-threshold movement; hard failures are untouchable.
    max_soft_threshold_delta: float = 0.03
    max_aggregate_soft_threshold_delta: float = 0.05
    hard_validator_bypass: bool = False
    case_can_mark_pass: bool = False
    case_can_mark_fail: bool = False
    #: Freshness half-life in seconds for the age-decay term (14 days).
    half_life_s: float = 14 * 24 * 3600.0
    #: Minimum labeled exposures before a retirement decision may be made.
    min_effective_exposures: float = 8.0
    #: Consecutive bad evaluation windows required for non-severe retirement.
    bad_windows_to_retire: int = 2

    def __post_init__(self):
        if self.hard_validator_bypass or self.case_can_mark_pass \
                or self.case_can_mark_fail:
            # Not a config choice. A policy that allowed any of these would
            # make the module's central invariant unenforceable, so it cannot
            # be constructed at all.
            raise ValueError(
                "case influence may never bypass a hard validator or decide a "
                "gate verdict; these flags exist to be asserted False")
        if self.max_single_case_delta > self.max_aggregate_delta:
            raise ValueError("single-case cap exceeds the aggregate cap")


DEFAULT_POLICIES = {
    STAGE_LOCATE: StageInfluencePolicy(stage=STAGE_LOCATE),
    STAGE_FAMILY: StageInfluencePolicy(stage=STAGE_FAMILY,
                                       max_single_case_delta=0.10,
                                       max_aggregate_delta=0.15,
                                       max_rank_shift=3),
    STAGE_GATE: StageInfluencePolicy(stage=STAGE_GATE,
                                     max_single_case_delta=0.03,
                                     max_aggregate_delta=0.05,
                                     max_rank_shift=0),
}


def policy_for(stage: str, policies: dict | None = None) -> StageInfluencePolicy:
    pol = (policies or DEFAULT_POLICIES).get(stage)
    if pol is None:
        raise KeyError(f"no influence policy for stage {stage!r}")
    return pol


# ================================================================== digests
def tree_digest(files: dict) -> str:
    """Content digest of a whole repo mapping ``{relpath: source}``.

    Uses ``snapshot.cse1_digest`` so this module and the snapshot spine cannot
    drift on what "the same tree" means.
    """
    return snap.cse1_digest({k: files[k] for k in sorted(files)})


def scope_digest(files: dict, paths) -> str:
    """Digest of ONLY the paths a case depends on (its evidence closure).

    A file the case never mentions can change all it likes without staling the
    case — that is the design's "only unrelated files changed -> preserve"
    row. Missing paths are recorded as ``None`` rather than skipped, so a
    DELETED dependency changes the digest instead of silently matching.
    """
    body = {p: files.get(p) for p in sorted(set(paths))}
    return snap.cse1_digest(body)


def freshness_state(case: dict, *, files: dict | None = None,
                    now: float | None = None,
                    policy: StageInfluencePolicy | None = None) -> str:
    """``fresh`` | ``aged`` | ``stale`` | ``unknown`` for one case.

    ``unknown`` when either the case or the caller cannot produce a digest —
    and ``unknown`` means zero influence, never "assume fine".
    """
    src = (case.get("freshness") or {}).get("source_scope_digest")
    if not src or files is None:
        return FRESHNESS_UNKNOWN
    paths = (case.get("scope") or {}).get("path_prefixes") or []
    if not paths:
        return FRESHNESS_UNKNOWN
    if scope_digest(files, paths) != src:
        return FRESHNESS_STALE
    pol = policy or DEFAULT_POLICIES[STAGE_LOCATE]
    age = max(0.0, (now if now is not None else time.time())
              - float((case.get("lifecycle") or {}).get("created_at") or 0.0))
    return FRESHNESS_FRESH if age < pol.half_life_s else FRESHNESS_AGED


def freshness_weight(state: str, case: dict, *, now: float | None = None,
                     policy: StageInfluencePolicy | None = None) -> float:
    """Multiplier in [0, 1]. ``stale`` and ``unknown`` are hard zeros."""
    if state in (FRESHNESS_STALE, FRESHNESS_UNKNOWN):
        return 0.0
    pol = policy or DEFAULT_POLICIES[STAGE_LOCATE]
    age = max(0.0, (now if now is not None else time.time())
              - float((case.get("lifecycle") or {}).get("created_at") or 0.0))
    return math.exp(-math.log(2.0) * age / pol.half_life_s)


# ================================================================== mining
def derive_verdict_id(row: dict) -> str:
    """Stable id for one autopsy verdict.

    ``probes/run_autopsy.py`` stamps no ``verdict_id`` (checked: the row keys
    are record/key/fn/kind/row_verdict/wall/detail/evidence/taxonomy/
    shape_problems/lever/cycle_id). The design assumes one, so it is DERIVED
    here from the three fields that do identify a verdict uniquely inside a
    board. If the autopsy ever emits a real id, ``_row_verdict_id`` prefers it.
    """
    return "%s:%s:%s" % (row.get("cycle_id", ""), row.get("key", ""),
                         row.get("wall", ""))


def _row_verdict_id(row: dict) -> str:
    return row.get("verdict_id") or derive_verdict_id(row)


def _tax(row: dict, cls: str):
    for t in row.get("taxonomy") or ():
        if isinstance(t, dict) and t.get("class") == cls:
            return t
    return None


def classify_row(row: dict) -> str | None:
    """Autopsy task row -> case type, or ``None`` when it earns no case.

    The only place the wall->type decision is made. ``locate_miss`` splits on
    the measured ``attractor_lock`` tag; without it, a locate_miss where gold
    WAS on the menu is a selection miss, and a locate_miss where gold was
    absent from the menu is neither (the union never carried it, so no ranking
    nudge could have helped — that is a vocabulary gap, a generator problem).
    """
    if row.get("record") != "task":
        return None
    wall = row.get("wall")
    if wall in UNMINEABLE_WALLS or wall not in WALL_CASE_CANDIDATES:
        return None
    if wall == "menu_miss":
        return CASE_FAMILY_MISS
    if wall == "gate_refusal":
        return CASE_GATE_REFUSAL
    if wall == "selection_miss":
        return CASE_SELECTION_MISS
    # locate_miss
    if _tax(row, "attractor_lock") is not None:
        return CASE_LOCATE_ATTRACTOR
    ev = row.get("evidence") or {}
    if ev.get("file_menu_gold_index") is not None:
        return CASE_SELECTION_MISS
    return None


#: Per type: the evidence keys a row MUST carry to state all five of the
#: design's mining preconditions (pattern / decision / why wrong / what signal
#: should have changed it / what proves it). A row missing any of these is a
#: RECORDED rejection, never a low-quality memory.
REQUIRED_EVIDENCE = {
    CASE_LOCATE_ATTRACTOR: ("gold_rel", "file_menu_pick", "file_menu_options"),
    CASE_SELECTION_MISS: ("gold_rel", "file_menu_pick", "file_menu_options",
                          "file_menu_gold_index"),
    CASE_GATE_REFUSAL: ("gold_rel", "refusing_gate", "refusal_correct_vs_gold"),
    CASE_FAMILY_MISS: ("gold_rel", "missing_family"),
}


def _require(ev: dict, case_type: str) -> None:
    missing = [k for k in REQUIRED_EVIDENCE[case_type]
               if ev.get(k) is None]
    if missing:
        raise MiningRefused(
            f"{case_type}: evidence missing {missing}; a case may not be mined "
            f"from an incomplete verdict")


def _canonical_body(case_type: str, row: dict) -> dict:
    """The normalized pattern a case is keyed on. NEVER includes ``cycle_id``.

    Excluding the cycle is what makes the same wall seen on two boards ONE case
    with two occurrences instead of two cases. Which fields survive is a
    deliberate, type-specific choice:

    * attractor — keyed on the HUB ALONE. The pattern being remembered is
      "this file keeps winning picks it should not"; it is a property of the
      repo, not of any one task, and keying it per task would shatter the one
      case that actually generalizes into N singletons. This is the one type
      whose occurrences legitimately span different tasks.
    * everything else — keyed on ``(task key, discriminator)``. A gate that
      refuses on one function is not evidence about another function, so
      "same wall + same task = one case" is the right grain, and re-running
      the same board folds into occurrence counts.
    """
    ev = row.get("evidence") or {}
    if case_type == CASE_LOCATE_ATTRACTOR:
        return {"t": case_type, "hub": _tax(row, "attractor_lock")["evidence"]["hub"]}
    if case_type == CASE_SELECTION_MISS:
        return {"t": case_type, "key": row.get("key"),
                "gold": ev.get("gold_rel"), "pick": ev.get("file_menu_pick")}
    if case_type == CASE_GATE_REFUSAL:
        return {"t": case_type, "key": row.get("key"),
                "gate": ev.get("refusing_gate")}
    return {"t": case_type, "key": row.get("key"),
            "family": ev.get("missing_family")}


def canonical_key(case_type: str, row: dict) -> str:
    return "sha256:" + snap.cse1_digest(_canonical_body(case_type, row)).split(":")[-1]


def _case_id(case_type: str, ckey: str) -> str:
    return "lcase-" + hashlib.sha256(
        (case_type + "|" + ckey).encode()).hexdigest()[:12]


def _scope_paths(case_type: str, row: dict) -> list:
    """The evidence closure: every path the case's claim actually depends on.

    Deliberately NOT "the whole file menu". For an attractor the claim is about
    the hub and the gold it displaced; for a gate refusal it is the file the
    gate refused on. Including the eight menu options would make the case go
    stale every time an unrelated candidate file changed, which is exactly the
    over-invalidation the design's "only unrelated files changed" row forbids.
    """
    ev = row.get("evidence") or {}
    paths = []
    if case_type == CASE_LOCATE_ATTRACTOR:
        tag = _tax(row, "attractor_lock")["evidence"]
        paths = [tag.get("hub"), tag.get("gold_rel") or ev.get("gold_rel")]
    elif case_type == CASE_SELECTION_MISS:
        paths = [ev.get("file_menu_pick"), ev.get("gold_rel")]
    elif case_type == CASE_GATE_REFUSAL:
        paths = [ev.get("gold_rel"), ev.get("located_file")]
    else:
        paths = [ev.get("gold_rel")]
    return sorted({p for p in paths if isinstance(p, str) and p})


def _trigger(case_type: str, row: dict) -> dict:
    ev = row.get("evidence") or {}
    if case_type == CASE_LOCATE_ATTRACTOR:
        tag = _tax(row, "attractor_lock")["evidence"]
        return {
            "attractor_path": tag.get("hub"),
            "attractor_picks_on_board": tag.get("hub_picks"),
            "displaced_gold": ev.get("gold_rel"),
            "gold_menu_index": ev.get("file_menu_gold_index"),
            "file_menu_margin": ev.get("file_menu_margin"),
            "located_file_source": ev.get("located_file_source"),
            "misleading_signals": [row.get("detail", "")],
        }
    if case_type == CASE_SELECTION_MISS:
        return {
            "picked_path": ev.get("file_menu_pick"),
            "gold_path": ev.get("gold_rel"),
            "gold_menu_index": ev.get("file_menu_gold_index"),
            "file_menu_margin": ev.get("file_menu_margin"),
            "file_menu_n": ev.get("file_menu_n"),
            "located_file_source": ev.get("located_file_source"),
        }
    if case_type == CASE_GATE_REFUSAL:
        return {
            "gate_id": ev.get("refusing_gate"),
            "refusal_correct_vs_gold": ev.get("refusal_correct_vs_gold"),
            "refused_path": ev.get("gold_rel"),
            "stages_reached": ev.get("stages_reached") or [],
            "spec_exec_verdict": ev.get("spec_exec_verdict"),
            "spec_exec_survivors": ev.get("spec_exec_survivors"),
            "mutations": ev.get("mutations"),
            "score_vector": ev.get("line_ballot_scores") or {},
            "line_ballot_decision": ev.get("line_ballot_decision"),
        }
    return {
        "missing_family": ev.get("missing_family"),
        "gold_shape": ev.get("gold_shape"),
        "gold_hunks": ev.get("gold_hunks"),
        "menu_n": ev.get("menu_n"),
        "gold_path": ev.get("gold_rel"),
        "block_holes": ev.get("block_holes") or {},
    }


def _guidance(case_type: str, trig: dict) -> dict:
    """Allow-listed structured effects ONLY.

    No thresholds, no code, no free text the runtime executes. Every value here
    is either a direction, a feature name, or a fallback naming what happens
    when the case is ignored — and the fallback is always "the baseline still
    runs".
    """
    if case_type == CASE_LOCATE_ATTRACTOR:
        return {
            "effect_kind": "downrank_lure_and_boost_target",
            "direction": "soft",
            "avoid_features": [trig["attractor_path"]],
            "target_features": [trig["displaced_gold"]],
            "discriminators": [
                "this path wins picks across unrelated tasks; require "
                "task-specific evidence before letting it take the pick",
            ],
            "fallback": "retain_baseline_location_in_beam",
        }
    if case_type == CASE_SELECTION_MISS:
        return {
            "effect_kind": "boost_target_within_menu",
            "direction": "soft",
            "avoid_features": [trig["picked_path"]],
            "target_features": [trig["gold_path"]],
            "discriminators": ["the gold was ON the menu and lost on score"],
            "fallback": "retain_baseline_location_in_beam",
        }
    if case_type == CASE_GATE_REFUSAL:
        return {
            "effect_kind": "retain_for_secondary_validation",
            "direction": "soften_refusal",
            "gate_id": trig["gate_id"],
            "discriminators": [
                "this gate has refused a verified-gold file before",
            ],
            "fallback": "human_review",
        }
    return {
        "effect_kind": "expand_family_alternatives",
        "direction": "soft",
        "consider_families": [trig["missing_family"]],
        "discriminators": [
            "this shape was not representable in the candidate menu",
        ],
        "fallback": "retain_baseline_family_as_secondary",
    }


def _summary(case_type: str, row: dict, trig: dict) -> str:
    if case_type == CASE_LOCATE_ATTRACTOR:
        return (f"{trig['attractor_path']} attracts the file pick; it displaced "
                f"{trig['displaced_gold']} on task {row.get('key')}")
    if case_type == CASE_SELECTION_MISS:
        return (f"gold {trig['gold_path']} sat at menu index "
                f"{trig['gold_menu_index']} and {trig['picked_path']} was "
                f"picked instead")
    if case_type == CASE_GATE_REFUSAL:
        return (f"gate {trig['gate_id']} refused on the gold file "
                f"{trig['refused_path']}")
    return (f"family {trig['missing_family']} was absent from the "
            f"{trig['menu_n']}-candidate menu for {trig['gold_path']}")


def _evidence_strength(case_type: str, row: dict, trig: dict) -> float:
    """A bounded [0.2, 0.95] confidence read off MEASURED fields only.

    Never a model score and never 1.0: a case is evidence, not proof, and the
    ceiling is what stops a single board from minting an unopposable case.
    """
    s = 0.5
    if case_type == CASE_LOCATE_ATTRACTOR:
        picks = trig.get("attractor_picks_on_board") or 0
        s = 0.45 + 0.1 * min(5, int(picks))          # more picks = more real
    elif case_type == CASE_SELECTION_MISS:
        m = trig.get("file_menu_margin")
        # A NARROW margin means the ranker nearly got it right, so a bounded
        # nudge is plausible. A wide margin means something bigger is wrong and
        # this case should not pretend a small delta would have fixed it.
        s = 0.75 if isinstance(m, (int, float)) and m < 0.5 else 0.4
    elif case_type == CASE_GATE_REFUSAL:
        s = 0.8 if trig.get("refusal_correct_vs_gold", "").startswith(
            "incorrect") else 0.25
    else:
        s = 0.7 if trig.get("gold_shape") else 0.5
    return round(max(0.2, min(0.95, s)), 4)


def _now() -> float:
    return time.time()


def build_case(row: dict, *, case_type: str | None = None,
               repo_id: str = "", files: dict | None = None,
               now: float | None = None) -> dict:
    """One autopsy task row -> one case record. Raises ``MiningRefused``.

    Pure: touches no store and no disk. ``files`` (the live repo the row was
    produced against) is OPTIONAL and its absence is recorded honestly as
    ``freshness.source_scope_digest = None``, which by policy means the case can
    never take live influence until a digest-carrying run revalidates it.
    """
    ct = case_type or classify_row(row)
    if ct is None:
        wall = row.get("wall")
        raise MiningRefused(
            UNMINEABLE_WALLS.get(wall)
            or f"wall {wall!r} yields no typed case for this row")
    ev = row.get("evidence") or {}
    _require(ev, ct)
    if ct == CASE_LOCATE_ATTRACTOR and _tax(row, "attractor_lock") is None:
        raise MiningRefused("attractor case requires the measured "
                            "attractor_lock taxonomy tag")

    trig = _trigger(ct, row)
    ckey = canonical_key(ct, row)
    paths = _scope_paths(ct, row)
    ts = now if now is not None else _now()
    return {
        "schema": SCHEMA,
        "case_id": _case_id(ct, ckey),
        "case_type": ct,
        "stage": CASE_STAGE[ct],
        "status": STATUS_PROBATION,
        "canonical_key": ckey,
        "summary": _summary(ct, row, trig),
        "scope": {
            "scope_level": "repository",
            "repo_id": repo_id,
            "path_prefixes": paths,
            "languages": ["python"],
        },
        "trigger": trig,
        "guidance": _guidance(ct, trig),
        "provenance": {
            "source_cycle_ids": [row.get("cycle_id", "")],
            "source_task_keys": [row.get("key", "")],
            "source_fns": [row.get("fn", "")],
            "source_verdict_ids": [_row_verdict_id(row)],
            # A REFERENCE into results/unified_gym.jsonl, not an invented id.
            "cert_refs": [{"cycle_id": row.get("cycle_id", ""),
                           "task_key": row.get("key", "")}],
            "levers": [row.get("lever", "")],
            "row_verdicts": [row.get("row_verdict", "")],
            "evidence_strength": _evidence_strength(ct, row, trig),
            "created_by": "run-autopsy",
        },
        "freshness": {
            "source_tree_digest": tree_digest(files) if files else None,
            "source_scope_digest": scope_digest(files, paths) if files else None,
        },
        "lifecycle": {
            "created_at": ts,
            "last_seen_at": ts,
            "last_validated_at": None,
            "retired_at": None,
            "retirement_reason": None,
            "occurrences": 1,
        },
        "stats": _new_stats(),
    }


def _new_stats() -> dict:
    return {
        "labeled_exposures": 0,
        "effective_exposures": 0.0,
        "hits": 0,
        "misses": 0,
        "neutral": 0,
        "harm": 0,
        "treatment_hits": 0,
        "treatment_labeled": 0,
        "control_hits": 0,
        "control_labeled": 0,
        "consecutive_bad_windows": 0,
        "last_evaluated_at": None,
        #: Index into the store's outcome stream at the last lifecycle
        #: evaluation. Everything after it is the CURRENT window.
        "window_cursor": 0,
    }


def _merge_occurrence(case: dict, row: dict, *, now: float | None = None) -> dict:
    """Fold a repeat sighting into an existing case. Idempotent per verdict id.

    Re-mining the SAME verdict must not inflate the occurrence count — that is
    what makes replaying a board safe, and it is the difference between "seen
    twice" and "read the file twice".
    """
    vid = _row_verdict_id(row)
    prov = case["provenance"]
    if vid in prov["source_verdict_ids"]:
        return case
    prov["source_verdict_ids"].append(vid)
    for field_name, value in (("source_cycle_ids", row.get("cycle_id", "")),
                              ("source_task_keys", row.get("key", "")),
                              ("source_fns", row.get("fn", "")),
                              ("levers", row.get("lever", "")),
                              ("row_verdicts", row.get("row_verdict", ""))):
        if value not in prov[field_name]:
            prov[field_name].append(value)
    ref = {"cycle_id": row.get("cycle_id", ""), "task_key": row.get("key", "")}
    if ref not in prov["cert_refs"]:
        prov["cert_refs"].append(ref)
    case["lifecycle"]["occurrences"] += 1
    case["lifecycle"]["last_seen_at"] = now if now is not None else _now()

    # An attractor case is keyed on the HUB ALONE, so a repeat sighting is
    # usually a DIFFERENT gold being displaced by the same lure. Recording only
    # the first one would leave the case downranking the hub while boosting a
    # target that is not even on this menu — a half-applied nudge, which is
    # worse than none. Every displaced gold joins target_features and the
    # evidence closure. (measured: utils/attempt_context.py displaces
    # background_workers.py AND routers/agent_tasks.py on the same boards.)
    if case["case_type"] == CASE_LOCATE_ATTRACTOR:
        gold = ((row.get("evidence") or {}).get("gold_rel")
                or (_tax(row, "attractor_lock") or {}).get("evidence", {}).get("gold_rel"))
        if isinstance(gold, str) and gold:
            tgt = case["guidance"].setdefault("target_features", [])
            if gold not in tgt:
                tgt.append(gold)
            also = case["trigger"].setdefault("also_displaced", [])
            if gold != case["trigger"].get("displaced_gold") and gold not in also:
                also.append(gold)
            paths = case["scope"]["path_prefixes"]
            if gold not in paths:
                # The closure grows, so the scope digest changes. A case
                # re-mined with more evidence is deliberately NOT still "fresh"
                # against its old digest; it must be revalidated.
                case["scope"]["path_prefixes"] = sorted(paths + [gold])
                case["freshness"]["source_scope_digest"] = None
    # Independent corroboration raises confidence, but the 0.95 ceiling in
    # _evidence_strength still holds: repetition is not proof.
    prov["evidence_strength"] = round(
        min(0.95, prov["evidence_strength"] + 0.03), 4)
    return case


# =============================================================== the store
@dataclass
class MiningReport:
    """What a mining run actually did. Every number is a count of real rows."""

    rows_seen: int = 0
    task_rows: int = 0
    mined: int = 0
    new_cases: int = 0
    merged: int = 0
    rejected: int = 0
    rejections: list = field(default_factory=list)
    by_type: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"rows_seen": self.rows_seen, "task_rows": self.task_rows,
                "mined": self.mined, "new_cases": self.new_cases,
                "merged": self.merged, "rejected": self.rejected,
                "by_type": dict(self.by_type),
                "rejections": self.rejections}


class LearnedCaseStore(cm.CaseStore):
    """Append-only learned-case base. Extends ``case_memory.CaseStore``.

    The base class owns ``load`` / ``_append`` / ``cases_file_hash`` and the
    solved-case ``op='case'``/``op='confirm'`` events; this subclass folds the
    learning ops on top. Nothing is ever rewritten: a retired case stays on
    disk and stays queryable, which is what makes an audit possible after a
    case has been excluded from influence.
    """

    def __init__(self, path=DEFAULT_PATH, *, load: bool = True,
                 policies: dict | None = None):
        self.learned: dict = {}
        self.consultations: list = []
        self.influences: list = []
        self.outcomes: list = []
        self.rejections: list = []
        self.policies = policies or DEFAULT_POLICIES
        super().__init__(path, load=load)

    # ------------------------------------------------------------- folding
    def load(self) -> None:
        self.learned = {}
        self.consultations = []
        self.influences = []
        self.outcomes = []
        self.rejections = []
        super().load()

    def _fold(self, ev: dict) -> None:
        op = ev.get("op")
        if op == "lcase":
            self.learned[ev["case"]["case_id"]] = json.loads(
                json.dumps(ev["case"]))
        elif op == "lobs":
            c = self.learned.get(ev["case_id"])
            if c is not None:
                c["provenance"] = ev["provenance"]
                c["lifecycle"] = ev["lifecycle"]
        elif op == "lstate":
            c = self.learned.get(ev["case_id"])
            if c is not None:
                c["status"] = ev["status"]
                c["lifecycle"].update(ev.get("lifecycle") or {})
                if ev.get("stats") is not None:
                    c["stats"] = ev["stats"]
        elif op == "consult":
            self.consultations.append(ev["record"])
        elif op == "influence":
            self.influences.append(ev["record"])
        elif op == "outcome":
            self.outcomes.append(ev["record"])
            c = self.learned.get(ev["record"]["case_id"])
            if c is not None and ev.get("stats") is not None:
                c["stats"] = ev["stats"]
        elif op == "reject":
            self.rejections.append(ev["record"])
        else:
            super()._fold(ev)

    # -------------------------------------------------------------- mining
    def mine_row(self, row: dict, *, repo_id: str = "",
                 files: dict | None = None, now: float | None = None):
        """Mine ONE autopsy task row. Returns the case, or ``None`` if refused.

        A refusal is RECORDED (``op='reject'``), never dropped: the design
        keeps rejected mining attempts so "why is there no case for this wall"
        has an answer on disk instead of in someone's memory.
        """
        try:
            case = build_case(row, repo_id=repo_id, files=files, now=now)
        except MiningRefused as exc:
            rec = {"verdict_id": _row_verdict_id(row), "wall": row.get("wall"),
                   "key": row.get("key"), "cycle_id": row.get("cycle_id"),
                   "reason": str(exc)}
            self.rejections.append(rec)
            self._append({"op": "reject", "record": rec})
            return None

        existing = self.learned.get(case["case_id"])
        if existing is None:
            self.learned[case["case_id"]] = case
            self._append({"op": "lcase", "case": case})
            return case
        _merge_occurrence(existing, row, now=now)
        self._append({"op": "lobs", "case_id": existing["case_id"],
                      "provenance": existing["provenance"],
                      "lifecycle": existing["lifecycle"]})
        return existing

    def mine_board(self, rows, *, repo_id: str = "", files: dict | None = None,
                   now: float | None = None) -> MiningReport:
        """Mine a whole autopsy board (the parsed jsonl rows)."""
        rep = MiningReport()
        for row in rows:
            rep.rows_seen += 1
            if row.get("record") != "task":
                continue
            rep.task_rows += 1
            before = len(self.learned)
            case = self.mine_row(row, repo_id=repo_id, files=files, now=now)
            if case is None:
                rep.rejected += 1
                rep.rejections.append(self.rejections[-1])
                continue
            rep.mined += 1
            rep.by_type[case["case_type"]] += 1
            if len(self.learned) > before:
                rep.new_cases += 1
            else:
                rep.merged += 1
        return rep

    def mine_files(self, paths, *, repo_id: str = "",
                   files: dict | None = None) -> MiningReport:
        """Mine one or more ``results/autopsy_*.jsonl`` boards from disk."""
        total = MiningReport()
        for p in paths:
            rows = load_board(p)
            rep = self.mine_board(rows, repo_id=repo_id, files=files)
            total.rows_seen += rep.rows_seen
            total.task_rows += rep.task_rows
            total.mined += rep.mined
            total.new_cases += rep.new_cases
            total.merged += rep.merged
            total.rejected += rep.rejected
            total.rejections.extend(rep.rejections)
            total.by_type.update(rep.by_type)
        return total

    # --------------------------------------------------------- consultation
    def consult(self, stage: str, context: "ConsultContext",
                *, now: float | None = None) -> "ConsultationBundle":
        """Retrieve bounded, staleness-aware evidence for ONE decision.

        Hard filters first (stage, repo, status), then structural matching,
        then ranking, then the policy's top-K and per-cluster limits. Returns a
        bundle whose every element carries its OWN influence bound; a caller
        that ignores the bounds has to do so deliberately.
        """
        pol = policy_for(stage, self.policies)
        results = []
        for case in self.learned.values():
            if case["stage"] != stage:
                continue
            if case["status"] in (STATUS_RETIRED,):
                continue
            if (case["scope"].get("repo_id") or "") != (context.repo_id or ""):
                continue
            sim = _similarity(case, context)
            if sim <= 0.0:
                continue
            fstate = freshness_state(case, files=context.files, now=now,
                                     policy=pol)
            fw = freshness_weight(fstate, case, now=now, policy=pol)
            trust = _learning_trust(case)
            prob = (pol.probation_multiplier
                    if case["status"] == STATUS_PROBATION else 1.0)
            weight = sim * case["provenance"]["evidence_strength"] * fw * trust * prob
            results.append(CaseHit(
                case=case, similarity=round(sim, 4), freshness=fstate,
                freshness_weight=round(fw, 4), learning_trust=round(trust, 4),
                probation_factor=prob, weight=round(weight, 6),
                # weight is already in [0, 1], so this is <= the per-case cap
                # by construction rather than by a clamp that could be removed.
                influence_bound=round(min(pol.max_single_case_delta,
                                          weight * pol.max_single_case_delta), 6),
                policy=pol))
        results.sort(key=lambda h: (-h.weight, h.case["case_id"]))
        results = _diversify(results, pol)
        bundle = ConsultationBundle(stage=stage, run_id=context.run_id,
                                    mode=context.mode, hits=results,
                                    policy=pol)
        self.record_consultation(bundle)
        return bundle

    def record_consultation(self, bundle: "ConsultationBundle") -> None:
        for rank, hit in enumerate(bundle.hits):
            rec = {"run_id": bundle.run_id, "case_id": hit.case["case_id"],
                   "stage": bundle.stage, "mode": bundle.mode,
                   "similarity": hit.similarity, "freshness": hit.freshness,
                   "learning_confidence": hit.learning_trust,
                   "influence_bound": hit.influence_bound,
                   "retrieval_rank": rank,
                   "eligible": bundle.mode == MODE_TREATMENT
                   and hit.weight > 0.0}
            self.consultations.append(rec)
            self._append({"op": "consult", "record": rec})

    def record_influence(self, run_id: str, ledger: "DecisionLedger") -> None:
        for rec in ledger.influence_records(run_id):
            self.influences.append(rec)
            self._append({"op": "influence", "record": rec})

    # ------------------------------------------------------------- outcomes
    def record_outcome(self, case_id: str, *, run_id: str, stage: str,
                       label: str, mode: str = MODE_TREATMENT,
                       severe: bool = False, weight: float = 1.0,
                       now: float | None = None) -> dict:
        """Label one exposure and fold it into the case's stats.

        ``neutral``/``unknown`` never count as misses. ``harm`` with
        ``severe=True`` retires the case immediately — the design's one
        no-quorum retirement, because a safety regression does not get a second
        window to prove itself.
        """
        if label not in OUTCOME_LABELS:
            raise ValueError(f"unknown outcome label {label!r}")
        if mode not in EVALUATION_MODES:
            raise ValueError(f"unknown evaluation mode {mode!r}")
        case = self.learned.get(case_id)
        if case is None:
            raise KeyError(case_id)
        st = case["stats"]
        if label != OUTCOME_UNKNOWN:
            st["labeled_exposures"] += 1
            st["effective_exposures"] = round(
                st["effective_exposures"] + max(0.0, float(weight)), 6)
        if label == OUTCOME_HIT:
            st["hits"] += 1
        elif label == OUTCOME_MISS:
            st["misses"] += 1
        elif label == OUTCOME_NEUTRAL:
            st["neutral"] += 1
        elif label == OUTCOME_HARM:
            st["harm"] += 1
        if label in (OUTCOME_HIT, OUTCOME_MISS, OUTCOME_HARM):
            bucket = "treatment" if mode == MODE_TREATMENT else "control"
            st[f"{bucket}_labeled"] += 1
            if label == OUTCOME_HIT:
                st[f"{bucket}_hits"] += 1
        st["last_evaluated_at"] = now if now is not None else _now()

        rec = {"case_id": case_id, "run_id": run_id, "stage": stage,
               "label": label, "evaluation_mode": mode, "severe": bool(severe),
               "weight": float(weight)}
        self.outcomes.append(rec)
        self._append({"op": "outcome", "record": rec, "stats": st})

        if label == OUTCOME_HARM and severe:
            self._set_status(case, STATUS_RETIRED,
                             reason="severe_safety_regression", now=now)
        return case

    def window_view(self, case_id: str) -> dict:
        """Stats over the outcomes recorded SINCE the last lifecycle evaluation.

        Deliberately NOT ``stats_view``. A lifetime cumulative rate can never
        recover from one bad stretch — a case that misbehaved early would stay
        condemned no matter how well it behaved afterwards, and the "two
        consecutive bad windows" rule would be meaningless because there is
        only ever one window. (Same trap as MC rule #160: a displayed rate must
        come from a trailing window over raw terminal events, never from an
        increment-only counter.) Lifetime numbers remain in ``stats_view`` for
        operator visibility; the LIFECYCLE decision reads this.
        """
        case = self.learned[case_id]
        cursor = int(case["stats"].get("window_cursor", 0))
        rows = [o for o in self.outcomes[cursor:] if o["case_id"] == case_id]
        t_hits = t_lab = c_hits = c_lab = harm = labeled = 0
        for o in rows:
            if o["label"] != OUTCOME_UNKNOWN:
                labeled += 1
            if o["label"] == OUTCOME_HARM:
                harm += 1
            if o["label"] not in (OUTCOME_HIT, OUTCOME_MISS, OUTCOME_HARM):
                continue
            if o["evaluation_mode"] == MODE_TREATMENT:
                t_lab += 1
                t_hits += o["label"] == OUTCOME_HIT
            else:
                c_lab += 1
                c_hits += o["label"] == OUTCOME_HIT
        t_rate = (t_hits / t_lab) if t_lab else None
        c_rate = (c_hits / c_lab) if c_lab else None
        return {
            "labeled": labeled,
            "treatment_success_rate": t_rate,
            "control_success_rate": c_rate,
            "lift": (t_rate - c_rate) if (t_rate is not None
                                          and c_rate is not None) else None,
            "harm_rate": (harm / labeled) if labeled else 0.0,
            "cursor": len(self.outcomes),
        }

    def evaluate_lifecycle(self, case_id: str, *,
                           now: float | None = None) -> dict:
        """Promote, hold, or retire ONE case from its CURRENT window.

        Ordering matters and is deliberate: a retired case is never revived
        here (the design requires a NEW verified observation for that), and
        promotion out of probation requires positive lift, not merely survival.
        """
        case = self.learned[case_id]
        if case["status"] == STATUS_RETIRED:
            return case
        pol = policy_for(case["stage"], self.policies)
        s = self.window_view(case_id)
        if s["labeled"] < pol.min_effective_exposures:
            return case                      # not enough evidence to judge
        bad = (s["lift"] is not None and s["lift"] < 0.0) or s["harm_rate"] > 0.0
        st = case["stats"]
        # The window closes here whether the verdict was good or bad; the next
        # evaluation judges only what happened after this moment.
        st["window_cursor"] = s["cursor"]
        if bad:
            st["consecutive_bad_windows"] += 1
            if st["consecutive_bad_windows"] >= pol.bad_windows_to_retire:
                return self._set_status(case, STATUS_RETIRED,
                                        reason="negative_lift_two_windows",
                                        now=now)
            self._append({"op": "lstate", "case_id": case_id,
                          "status": case["status"], "stats": st,
                          "lifecycle": case["lifecycle"]})
            return case
        st["consecutive_bad_windows"] = 0
        if case["status"] == STATUS_PROBATION and (s["lift"] or 0.0) > 0.0:
            return self._set_status(case, STATUS_ACTIVE, reason=None, now=now)
        self._append({"op": "lstate", "case_id": case_id,
                      "status": case["status"], "stats": st,
                      "lifecycle": case["lifecycle"]})
        return case

    def mark_stale(self, case_id: str, *, now: float | None = None) -> dict:
        """Exclude from live influence WITHOUT deleting. Reversible by revalidation."""
        return self._set_status(self.learned[case_id], STATUS_STALE,
                                reason="scope_digest_changed", now=now)

    def revalidate(self, case_id: str, files: dict, *,
                   now: float | None = None) -> dict:
        """Re-anchor a stale case to the CURRENT tree.

        Creates a new digest observation rather than rewriting the old evidence,
        and drops the case back to probation: a re-anchored case has to earn its
        trust again against code it has never actually been tested on.
        """
        case = self.learned[case_id]
        if case["status"] == STATUS_RETIRED:
            raise ValueError("a retired case is never revived by revalidation; "
                             "mine a successor case from a fresh verdict")
        paths = case["scope"]["path_prefixes"]
        case["freshness"] = {"source_tree_digest": tree_digest(files),
                             "source_scope_digest": scope_digest(files, paths)}
        ts = now if now is not None else _now()
        case["lifecycle"]["last_validated_at"] = ts
        case["lifecycle"]["created_at"] = ts        # age decay restarts here
        return self._set_status(case, STATUS_PROBATION, reason=None, now=ts)

    def _set_status(self, case: dict, status: str, *, reason: str | None,
                    now: float | None = None) -> dict:
        case["status"] = status
        ts = now if now is not None else _now()
        if status == STATUS_RETIRED:
            case["lifecycle"]["retired_at"] = ts
            case["lifecycle"]["retirement_reason"] = reason
        self._append({"op": "lstate", "case_id": case["case_id"],
                      "status": status, "lifecycle": case["lifecycle"],
                      "stats": case["stats"]})
        return case

    # ---------------------------------------------------------------- views
    def case_table(self) -> list:
        """Sorted, human-readable table of every mined case."""
        rows = []
        for c in self.learned.values():
            rows.append({
                "case_id": c["case_id"], "case_type": c["case_type"],
                "stage": c["stage"], "status": c["status"],
                "occurrences": c["lifecycle"]["occurrences"],
                "cycles": len(c["provenance"]["source_cycle_ids"]),
                "tasks": len(c["provenance"]["source_task_keys"]),
                "evidence": c["provenance"]["evidence_strength"],
                "summary": c["summary"],
            })
        rows.sort(key=lambda r: (r["case_type"], -r["occurrences"],
                                 r["case_id"]))
        return rows


# ============================================================ retrieval math
def _similarity(case: dict, context: "ConsultContext") -> float:
    """Structured-feature similarity in [0, 1]. No embeddings (see OUT_OF_SCOPE).

    Zero means "this case has nothing to say here" and short-circuits the whole
    consultation, which is why every branch below returns 0.0 rather than a
    small positive floor: a case that does not match must contribute NOTHING,
    not a little.
    """
    ct = case["case_type"]
    trig = case["trigger"]
    if ct == CASE_LOCATE_ATTRACTOR:
        if trig["attractor_path"] not in (context.candidate_paths or ()):
            return 0.0
        # Strongest when the known lure is on the menu AND the known displaced
        # target is too — that is the exact configuration the case saw.
        return 1.0 if trig["displaced_gold"] in (context.candidate_paths or ()) \
            else 0.6
    if ct == CASE_SELECTION_MISS:
        if context.task_key and context.task_key != _first(
                case["provenance"]["source_task_keys"]):
            return 0.0
        paths = set(context.candidate_paths or ())
        if trig["picked_path"] not in paths:
            return 0.0
        return 1.0 if trig["gold_path"] in paths else 0.5
    if ct == CASE_GATE_REFUSAL:
        if context.gate_id and context.gate_id != trig["gate_id"]:
            return 0.0
        if context.task_key and context.task_key != _first(
                case["provenance"]["source_task_keys"]):
            return 0.5 if context.gate_id else 0.0
        return 1.0
    # family_miss
    if context.task_key and context.task_key != _first(
            case["provenance"]["source_task_keys"]):
        return 0.0
    fams = set(context.enabled_families or ())
    # A family already enabled means the gap is closed; the case is not needed.
    return 0.0 if trig["missing_family"] in fams else 1.0


def _first(seq):
    return seq[0] if seq else None


def _learning_trust(case: dict) -> float:
    """Laplace-smoothed posterior hit rate in [0.1, 1.0].

    Smoothing matters: an unproven case must not start at 0 (it would never get
    an exposure to prove itself) nor at 1 (it would start with full authority).
    Two pseudo-counts put a brand-new case at 0.5 and require real evidence to
    move it either way.
    """
    st = case["stats"]
    hits = st["hits"]
    tried = st["hits"] + st["misses"] + st["harm"]
    return max(0.1, min(1.0, (hits + 1.0) / (tried + 2.0)))


def stats_view(case: dict) -> dict:
    """Derived learning statistics. Recomputable from the append-only events."""
    st = case["stats"]
    tried = st["hits"] + st["misses"] + st["harm"]
    labeled = st["labeled_exposures"]
    t_lab, c_lab = st["treatment_labeled"], st["control_labeled"]
    t_rate = (st["treatment_hits"] / t_lab) if t_lab else None
    c_rate = (st["control_hits"] / c_lab) if c_lab else None
    return {
        "labeled_exposures": labeled,
        "effective_exposures": st["effective_exposures"],
        "hit_rate": (st["hits"] / tried) if tried else None,
        "posterior_hit_rate": _learning_trust(case),
        "treatment_success_rate": t_rate,
        "control_success_rate": c_rate,
        # Lift is only defined against a matched control. Without one it is
        # None, NOT the raw hit rate — conflating them is how a case that was
        # only ever asked easy questions gets promoted.
        "lift": (t_rate - c_rate) if (t_rate is not None
                                      and c_rate is not None) else None,
        "harm_rate": (st["harm"] / labeled) if labeled else 0.0,
        "consecutive_bad_windows": st["consecutive_bad_windows"],
    }


def _cluster_key(hit: "CaseHit") -> tuple:
    """Cases from ONE source cycle+type form a cluster.

    Without this, one bad board contributes six near-identical cases to a single
    decision and blows straight through the per-case cap by arithmetic.
    """
    prov = hit.case["provenance"]
    return (hit.case["case_type"], _first(prov["source_cycle_ids"]))


def _diversify(hits: list, pol: StageInfluencePolicy) -> list:
    seen: Counter = Counter()
    out = []
    for h in hits:
        k = _cluster_key(h)
        if seen[k] >= pol.max_cases_per_cluster:
            continue
        seen[k] += 1
        out.append(h)
        if len(out) >= pol.max_cases_per_decision:
            break
    return out


# ================================================================ dataclasses
@dataclass(frozen=True)
class ConsultContext:
    """Everything a consultation may look at. Nothing else is readable."""

    run_id: str
    repo_id: str = ""
    mode: str = MODE_TREATMENT
    task_key: str | None = None
    candidate_paths: tuple = ()
    enabled_families: tuple = ()
    gate_id: str | None = None
    files: dict | None = None


@dataclass(frozen=True)
class CaseHit:
    case: dict
    similarity: float
    freshness: str
    freshness_weight: float
    learning_trust: float
    probation_factor: float
    weight: float
    influence_bound: float
    policy: StageInfluencePolicy

    @property
    def case_id(self) -> str:
        return self.case["case_id"]

    def signed_delta(self, path: str) -> float:
        """Bounded delta this case proposes for ONE candidate path.

        Positive boosts, negative downranks, zero means "no opinion". The
        magnitude is already clamped by ``influence_bound``; the sign comes
        from the allow-listed guidance, never from case prose.
        """
        g = self.case["guidance"]
        if path in (g.get("target_features") or ()):
            return +self.influence_bound
        if path in (g.get("avoid_features") or ()):
            return -self.influence_bound
        return 0.0


@dataclass
class ConsultationBundle:
    stage: str
    run_id: str
    mode: str
    hits: list
    policy: StageInfluencePolicy

    @property
    def applies(self) -> bool:
        """Control and shadow buckets CONSULT and LOG but never apply."""
        return self.mode == MODE_TREATMENT

    def eligible(self) -> list:
        """Hits that may actually move anything (fresh, non-retired, treatment)."""
        if not self.applies:
            return []
        return [h for h in self.hits if h.weight > 0.0]


@dataclass
class DecisionLedger:
    """Baseline AND adjusted, side by side, for every influenced decision.

    The design requires both. A ledger that recorded only the adjusted result
    would make "did the case help?" unanswerable after the fact, which is the
    failure mode the whole evaluation half exists to prevent.
    """

    stage: str
    run_id: str
    mode: str
    baseline: list = field(default_factory=list)
    adjusted: list = field(default_factory=list)
    deltas: dict = field(default_factory=dict)
    per_case: list = field(default_factory=list)

    def influence_records(self, run_id: str) -> list:
        out = []
        for row in self.per_case:
            out.append({"run_id": run_id, "stage": self.stage,
                        "mode": self.mode, **row})
        return out

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "stage": self.stage,
                "evaluation_mode": self.mode,
                "baseline": {"order": list(self.baseline)},
                "adjusted": {"order": list(self.adjusted)},
                "deltas": dict(self.deltas),
                "cases": list(self.per_case)}


# ============================================================ influence apply
def apply_rank_adjustments(baseline: list, bundle: ConsultationBundle,
                           *, run_id: str | None = None) -> DecisionLedger:
    """Bounded reordering of a candidate list. Returns a PERMUTATION, always.

    Three structural guarantees, each of which a test pins directly:

    1. **Nothing is removed.** The output is built by sorting the input, so
       ``set(adjusted) == set(baseline)`` by construction.
    2. **Nothing moves further than the policy allows.** After sorting by
       adjusted score, each element is clamped back to within
       ``max_rank_shift`` of its baseline index.
    3. **The baseline top candidate survives.** It cannot fall below
       ``max_rank_shift``, so it is always still in any beam of that width.

    In control/shadow mode the adjusted order is IDENTICAL to the baseline and
    the deltas are still recorded — that is the control arm, and it is what
    makes lift measurable rather than assumed.
    """
    pol = bundle.policy
    ledger = DecisionLedger(stage=bundle.stage, run_id=run_id or bundle.run_id,
                            mode=bundle.mode, baseline=list(baseline))
    hits = bundle.eligible()
    raw: dict = {p: 0.0 for p in baseline}
    for hit in hits:
        for path in baseline:
            d = hit.signed_delta(path)
            if d == 0.0:
                continue
            raw[path] += d
            ledger.per_case.append({
                "case_id": hit.case_id, "candidate": path,
                "similarity": hit.similarity, "proposed_delta": round(d, 6),
                "case_cap": pol.max_single_case_delta,
                "aggregate_cap": pol.max_aggregate_delta,
                "freshness": hit.freshness,
                "attribution_mode": bundle.mode})
    # Aggregate clamp: no coalition of cases exceeds the stage-wide cap.
    deltas = {p: max(-pol.max_aggregate_delta,
                     min(pol.max_aggregate_delta, v))
              for p, v in raw.items()}
    ledger.deltas = {p: round(v, 6) for p, v in deltas.items() if v}

    if not bundle.applies:
        ledger.adjusted = list(baseline)
        return ledger

    base_index = {p: i for i, p in enumerate(baseline)}
    # Score = position (lower is better) minus the boost. Ties break on the
    # baseline order so the transformation is deterministic.
    ordered = sorted(baseline,
                     key=lambda p: (base_index[p] - deltas[p] * len(baseline),
                                    base_index[p]))
    ledger.adjusted = _clamp_rank_shift(baseline, ordered, pol.max_rank_shift)
    for row in ledger.per_case:
        row["baseline_rank"] = base_index[row["candidate"]]
        row["adjusted_rank"] = ledger.adjusted.index(row["candidate"])
        row["applied_delta"] = ledger.deltas.get(row["candidate"], 0.0)
    return ledger


def _clamp_rank_shift(baseline: list, ordered: list, max_shift: int) -> list:
    """Force every element back within ``max_shift`` of its baseline index.

    Implemented as a repair loop rather than a constrained sort because the
    constraint is per-element and a single pass can leave a violation behind.
    It terminates: each swap strictly reduces total displacement, and the
    baseline order is a fixed point.
    """
    if max_shift <= 0:
        return list(baseline)
    base_index = {p: i for i, p in enumerate(baseline)}
    cur = list(ordered)
    for _ in range(len(cur) * len(cur) + 1):
        bad = None
        for i, p in enumerate(cur):
            if abs(i - base_index[p]) > max_shift:
                bad = (i, p)
                break
        if bad is None:
            break
        i, p = bad
        target = base_index[p] + (max_shift if i > base_index[p] else -max_shift)
        target = max(0, min(len(cur) - 1, target))
        cur.pop(i)
        cur.insert(target, p)
    return cur


@dataclass(frozen=True)
class GateAdvice:
    """The strongest thing a gate case may ever say.

    ``disposition`` is drawn from ``GATE_DISPOSITIONS``, which contains no
    passing verdict. ``soft_threshold_delta`` is capped and only ever softens.
    Hard failures are not an input here at all — ``gate_advice`` refuses to
    return anything but ``no_effect`` when one is present, because there is no
    representable way for a case to touch them.
    """

    disposition: str
    soft_threshold_delta: float
    cases: tuple
    reason: str

    def __post_init__(self):
        if self.disposition not in GATE_DISPOSITIONS:
            raise ValueError(f"disposition {self.disposition!r} is not in the "
                             f"gate allow-list {GATE_DISPOSITIONS}")
        if self.soft_threshold_delta < 0.0:
            raise ValueError("a gate case may only SOFTEN a soft threshold")


def gate_advice(bundle: ConsultationBundle, *, hard_failures=()) -> GateAdvice:
    """Bounded gate consultation. Never passes, never fails, never skips.

    ``hard_failures`` is any non-empty collection of hard validator failures
    (compile / regression / security / required). Its presence short-circuits
    to ``no_effect`` BEFORE any case is read, so a case cannot even be asked to
    weigh in on a hard failure.
    """
    if hard_failures:
        return GateAdvice(disposition="no_effect", soft_threshold_delta=0.0,
                          cases=(), reason="hard validator failure is not "
                                           "influenceable by case memory")
    pol = bundle.policy
    hits = bundle.eligible()
    if not hits:
        return GateAdvice(disposition="no_effect", soft_threshold_delta=0.0,
                          cases=(), reason="no eligible gate cases")
    delta = 0.0
    for hit in hits:
        delta += min(pol.max_soft_threshold_delta,
                     hit.weight * pol.max_soft_threshold_delta)
    delta = min(delta, pol.max_aggregate_soft_threshold_delta)
    # Design: with no secondary validator the answer is needs_review, never
    # pass. We do not know the caller's validator inventory, so the STRONGEST
    # disposition available is "keep it alive for a second opinion".
    return GateAdvice(disposition="retain_for_secondary_validation",
                      soft_threshold_delta=round(delta, 6),
                      cases=tuple(h.case_id for h in hits),
                      reason="a verified-gold file was refused by this gate before")


@dataclass(frozen=True)
class FamilyAdvice:
    """Families to ADD. There is no removal field, by construction."""

    consider_families: tuple
    keep_baseline: bool
    cases: tuple

    def apply(self, baseline_families) -> tuple:
        """Union, baseline first. A family can never leave the set this way."""
        out = list(baseline_families)
        for f in self.consider_families:
            if f not in out:
                out.append(f)
        return tuple(out)


def family_advice(bundle: ConsultationBundle) -> FamilyAdvice:
    fams, cases = [], []
    for hit in bundle.eligible():
        for f in hit.case["guidance"].get("consider_families") or ():
            if f and f not in fams:
                fams.append(f)
        cases.append(hit.case_id)
    return FamilyAdvice(consider_families=tuple(fams), keep_baseline=True,
                        cases=tuple(cases))


# ==================================================================== boards
def load_board(path) -> list:
    """Parse one ``results/autopsy_*.jsonl`` board, stamping ``cycle_id``.

    A corrupt line is SKIPPED (same discipline as ``case_memory.load``): one bad
    byte must not cost the whole board, and the count is recoverable from
    ``rows_seen`` vs the file's line count.
    """
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ===================================================================== seams
#: THE THREE CONSULTATION POINTS IN ``fixer.py``.
#:
#: Documented, not wired. Each entry names the EXACT function, what it would
#: receive, and the invariant the integration must preserve. Nothing in this
#: module imports or edits ``fixer.py``; wiring is a separate, human-reviewed
#: change, and until it happens every case here is inert by construction.
SEAMS = {
    STAGE_LOCATE: {
        "function": "fixer.rank_file_menu",
        "signature": ("rank_file_menu(cand, func_evidence, idx, task_text, "
                      "fn_anchor_files, cfg, fan_in=None, files=None, "
                      "idf=None)"),
        "file": "fixer.py",
        "line": 1217,
        "consumes": [CASE_LOCATE_ATTRACTOR, CASE_SELECTION_MISS],
        "call": ("build ConsultContext(candidate_paths=tuple(cand), "
                 "repo_id=..., files=files) -> store.consult('locate', ctx) -> "
                 "apply_rank_adjustments(list(cand), bundle)"),
        "invariant": ("rank_file_menu's docstring already promises a "
                      "PERMUTATION of `cand`; apply_rank_adjustments returns a "
                      "permutation, so the existing contract is preserved "
                      "exactly. The baseline top candidate cannot fall below "
                      "max_rank_shift, so the current pick stays in any beam."),
        "guard": ("the fan_in_penalty precedent applies: the integration must "
                  "be provably INERT when no case matches — an empty bundle "
                  "returns ledger.adjusted == baseline, byte-identical."),
    },
    STAGE_FAMILY: {
        "function": "fixer.build_mutation_menu",
        "signature": ("build_mutation_menu(span, lo, cls, lines, fam_enabled, "
                      "providers=None)"),
        "file": "fixer.py",
        "line": 215,
        "consumes": [CASE_FAMILY_MISS],
        "call": ("ConsultContext(enabled_families=tuple(fam_enabled)) -> "
                 "store.consult('family_select', ctx) -> "
                 "family_advice(bundle).apply(fam_enabled)"),
        "invariant": ("FamilyAdvice.apply is a UNION with the baseline first, "
                      "so arm 3's round-robin ordering is untouched for every "
                      "family that was already enabled and a suggested family "
                      "can only ever be appended."),
        "guard": ("a family the case names but the providers cannot build is a "
                  "no-op, not an error; build_mutation_menu already tolerates "
                  "a family that produces nothing."),
    },
    STAGE_GATE: {
        "function": "fixer.run",
        "signature": "run(root, task_text, apply=False, cfg=None, ...)",
        "file": "fixer.py",
        "line": 4755,
        "refusal_sites": [
            {"line": 6020, "gate": "spec_exec:SHORTLIST"},
            {"line": 6463, "gate": "line_emission_rejected"},
            {"line": 6475, "gate": "line_emission:ladder_exhausted"},
        ],
        "consumes": [CASE_GATE_REFUSAL],
        "call": ("at the moment a refusal is about to be recorded: "
                 "ConsultContext(gate_id=<the refusing gate string>) -> "
                 "store.consult('gate', ctx) -> gate_advice(bundle, "
                 "hard_failures=...) -> record advice on the cert; the gate "
                 "still decides."),
        "invariant": ("gate_advice can return only retain_for_secondary_"
                      "validation / human_review / no_effect. There is no code "
                      "path by which it returns a pass, and hard_failures "
                      "short-circuits before any case is consulted."),
        "guard": ("the refusal must still be STAMPED on the certificate even "
                  "when a case advises retention — otherwise the next autopsy "
                  "loses the very evidence this case was mined from."),
    },
}


def seam_report() -> str:
    """Human-readable summary of the three seams (for a wiring review)."""
    out = []
    for stage, s in SEAMS.items():
        out.append(f"{stage}: {s['function']}  ({s['file']}:{s['line']})")
        out.append(f"    consumes : {', '.join(s['consumes'])}")
        out.append(f"    call     : {s['call']}")
        out.append(f"    invariant: {s['invariant']}")
    return "\n".join(out)


__all__ = [
    "CASE_FAMILY_MISS", "CASE_GATE_REFUSAL", "CASE_LOCATE_ATTRACTOR",
    "CASE_SELECTION_MISS", "CASE_STAGE", "CASE_TYPES", "DEFAULT_PATH",
    "DEFAULT_POLICIES", "EVALUATION_MODES", "FRESHNESS_AGED", "FRESHNESS_FRESH",
    "FRESHNESS_STALE", "FRESHNESS_UNKNOWN", "GATE_DISPOSITIONS",
    "MODE_CONTROL", "MODE_REPLAY", "MODE_SHADOW", "MODE_TREATMENT",
    "NEVER_SKIP_VALIDATORS", "OUTCOME_HARM", "OUTCOME_HIT", "OUTCOME_LABELS",
    "OUTCOME_MISS", "OUTCOME_NEUTRAL", "OUTCOME_UNKNOWN", "OUT_OF_SCOPE",
    "SCHEMA", "SEAMS", "STAGE_FAMILY", "STAGE_GATE", "STAGE_LOCATE",
    "STATUS_ACTIVE", "STATUS_PROBATION", "STATUS_RETIRED", "STATUS_STALE",
    "UNMINEABLE_WALLS", "WALL_CASE_CANDIDATES",
    "CaseHit", "ConsultContext", "ConsultationBundle", "DecisionLedger",
    "FamilyAdvice", "GateAdvice", "LearnedCaseStore", "MiningRefused",
    "MiningReport", "StageInfluencePolicy",
    "apply_rank_adjustments", "build_case", "canonical_key", "classify_row",
    "derive_verdict_id", "family_advice", "freshness_state",
    "freshness_weight", "gate_advice", "load_board", "policy_for",
    "scope_digest", "seam_report", "stats_view", "tree_digest",
]
