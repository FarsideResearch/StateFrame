"""Rank recent source changes as possible causes of a reported failure."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_GIT_TIMEOUT = 120

# ============================================================ tunables (v1)
#: Half-lives in DEPLOYMENT-INTERVAL units (here: commit-window position,
#: since no real deployment timestamps are wired — doc §6 names this
#: fallback explicitly: "deployment-count age is usually preferable").
HALF_LIFE_DIRECT = 4.0
HALF_LIFE_COCHANGE = 10.0

MIN_COCHANGE_SUPPORT = 2
MAX_FILES_PER_COMMIT = 40          # bulk/vendor sweep guard, mirrors history.py

#: Bounded-evidence caps (doc §9 "Every feature contribution is capped").
CAP_FEATURE = 1.0
CAP_TOTAL_E = 3.0

#: Content gate (doc §9). floor/reference bound the linear ramp; a candidate
#: below floor gets (approximately) zero git adjustment no matter how hot.
CONTENT_FLOOR = 0.05
CONTENT_REFERENCE = 0.9

DEFAULT_WEIGHTS = {
    "blame_overlap": 1.0,
    "changed_in_window": 0.6,
    "cochange_coupling": 0.5,
    "churn_surprise": 0.7,
}


# ==================================================================== git IO
def git(root: str, *args: str) -> str:
    """Read-only git, same shape as graphs.history.git: no shell, timeout,
    stdout only. Non-zero exit -> '' (an empty, never fabricated, history)."""
    r = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )
    return r.stdout if r.returncode == 0 else ""


@dataclass(frozen=True)
class CommitRecord:
    sha: str
    files: Tuple[str, ...]


def walk_commits_from(root: str, anchor: str, subtree: str,
                       window: int) -> List[CommitRecord]:
    """Newest-first commits reachable from ``anchor`` touching ``subtree``.

    Anchoring at an arbitrary ref (not HEAD) is the piece graphs.history.py
    deliberately does not offer (its window is always HEAD-relative) and
    calibration needs: it must replay against the snapshot available AT the
    fix, per the doc's §3 "snapshot discipline" (never include the fix
    itself — callers anchor at ``<fix_sha>^``, one commit before the fix).
    """
    raw = git(root, "log", anchor, f"-n{window}", "--no-merges",
              "--name-only", "--pretty=format:%x01%H", "--", subtree)
    out: List[CommitRecord] = []
    for chunk in raw.split("\x01"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = chunk.split("\n")
        sha = lines[0].strip()
        files = tuple(sorted({ln.strip() for ln in lines[1:] if ln.strip()}))
        if sha:
            out.append(CommitRecord(sha=sha, files=files))
    return out


def is_git_repo(root: str) -> bool:
    return os.path.isdir(os.path.join(root, ".git"))


# ============================================================= feature math
def decay(age_idx: int, half_life: float) -> float:
    """Exponential half-life decay over commit-window position (doc §6).
    age_idx 0 == most recent commit in the window -> full weight."""
    if age_idx < 0:
        return 0.0
    return 2.0 ** (-age_idx / half_life)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _logit(p: float, eps: float = 1e-4) -> float:
    p = _clip(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


def most_recent_touch_idx(path: str, commits: Sequence[CommitRecord]) -> Optional[int]:
    """Position (0 = newest) of the most recent commit in ``commits`` that
    touches ``path``, or None if the file never appears in the window."""
    for i, c in enumerate(commits):
        if path in c.files:
            return i
    return None


def blame_overlap(path: str, commits: Sequence[CommitRecord],
                   half_life: float = HALF_LIFE_DIRECT) -> float:
    """v1 file-level proxy for doc §7.1: decay of the most recent commit that
    touched this file within the window. 1.0 == touched in the newest commit."""
    idx = most_recent_touch_idx(path, commits)
    return 0.0 if idx is None else decay(idx, half_life)


def changed_in_window(path: str, commits: Sequence[CommitRecord]) -> float:
    """Binary direct-change feature (doc §7.2's ``I_changed``), file-grain."""
    return 1.0 if most_recent_touch_idx(path, commits) is not None else 0.0


def _cochange_pairs(commits: Sequence[CommitRecord]) -> Tuple[Counter, Dict[str, int]]:
    pairs: Counter = Counter()
    touches: Dict[str, int] = Counter()
    for c in commits:
        files = [f for f in c.files]
        if len(files) > MAX_FILES_PER_COMMIT:
            # Bulk sweep: counts toward nothing but its own churn elsewhere;
            # never toward co-change (would make every file "coupled").
            continue
        for f in files:
            touches[f] += 1
        for i, a in enumerate(files):
            for b in files[i + 1:]:
                key = (a, b) if a < b else (b, a)
                pairs[key] += 1
    return pairs, dict(touches)


def cochange_coupling(target: str, changed_files: Iterable[str],
                       commits: Sequence[CommitRecord],
                       min_support: int = MIN_COCHANGE_SUPPORT,
                       half_life: float = HALF_LIFE_COCHANGE) -> float:
    """doc §7.3 ``K(e, W)``: indirect coupling of ``target`` to the set of
    files DIRECTLY changed in the window, excluding target's own direct
    touches (those are ``blame_overlap``/``changed_in_window``'s job)."""
    pairs, touches = _cochange_pairs(commits)
    target_touches = touches.get(target, 0)
    if target_touches == 0:
        return 0.0
    total = 0.0
    for f in changed_files:
        if f == target:
            continue
        key = (target, f) if target < f else (f, target)
        support = pairs.get(key, 0)
        if support < min_support:
            continue
        # normalized association, residualized against how often `target`
        # co-occurs with ANYTHING (shrinks hub files toward zero).
        assoc = support / float(target_touches)
        idx = most_recent_touch_idx(f, commits)
        w = decay(idx, half_life) if idx is not None else 0.0
        total += assoc * w
    return _clip(total, 0.0, CAP_FEATURE)


def churn_adjusted_surprise(path: str, recent: Sequence[CommitRecord],
                             baseline: Sequence[CommitRecord]) -> float:
    """doc §7.4 residual: observed recent touch rate minus the file's own
    longer-baseline touch rate, standardized and clipped. A file that always
    changes should NOT read as surprising just because it changed again."""
    if not recent or not baseline:
        return 0.0
    recent_rate = sum(1 for c in recent if path in c.files) / float(len(recent))
    baseline_rate = sum(1 for c in baseline if path in c.files) / float(len(baseline))
    # Standard deviation of a Bernoulli-ish touch process; epsilon guards a
    # file with zero baseline variance from producing an unbounded residual.
    sigma = math.sqrt(max(baseline_rate * (1.0 - baseline_rate), 1e-6))
    r = (recent_rate - baseline_rate) / sigma
    return _clip(r, -CAP_FEATURE, CAP_FEATURE)


# ================================================================ candidate
@dataclass(frozen=True)
class Candidate:
    entity: str                       # file path (v1 grain)
    content_score: Optional[float] = None   # None -> uniform prior (1.0)

    @property
    def prior(self) -> float:
        return 1.0 if self.content_score is None else self.content_score


@dataclass(frozen=True)
class GitEvidence:
    blame_overlap: float
    changed_in_window: float
    cochange_coupling: float
    churn_adjusted_surprise: float
    window_confidence: float


@dataclass(frozen=True)
class RankedSuspect:
    entity: str
    score: float
    content_score: float
    git_evidence: GitEvidence
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def content_gate(content_prob: float, floor: float = CONTENT_FLOOR,
                  reference: float = CONTENT_REFERENCE) -> float:
    """doc §9 ``gate(e)``: the churn-trap guard. A candidate the content prior
    considers implausible gets a git adjustment near zero regardless of how
    hot or coupled it is."""
    if reference <= floor:
        return 1.0 if content_prob >= reference else 0.0
    return _clip((content_prob - floor) / (reference - floor), 0.0, 1.0)


def rank_suspects(candidates: Sequence[Candidate],
                   commits: Sequence[CommitRecord],
                   weights: Mapping[str, float] = DEFAULT_WEIGHTS,
                   window_confidence: float = 1.0,
                   min_cochange_support: int = MIN_COCHANGE_SUPPORT,
                   baseline: Optional[Sequence[CommitRecord]] = None,
                   ) -> List[RankedSuspect]:
    """doc §10 ``rank_suspects``, file-level, single-window (Case C style —
    the caller resolves window selection; see module docstring). Content
    score is the ranking PRIOR; git evidence is a bounded LIKELIHOOD
    adjustment, never a replacement (§9) — that is the churn-trap guard.
    """
    changed_files = [c.sha and f for c in commits for f in c.files]  # flat, may repeat
    changed_set = {f for c in commits for f in c.files}
    out: List[RankedSuspect] = []
    for cand in candidates:
        b = blame_overlap(cand.entity, commits)
        x = changed_in_window(cand.entity, commits)
        k = cochange_coupling(cand.entity, changed_set, commits,
                               min_support=min_cochange_support)
        r = churn_adjusted_surprise(cand.entity, commits, baseline or commits)
        ev = GitEvidence(blame_overlap=b, changed_in_window=x,
                          cochange_coupling=k, churn_adjusted_surprise=r,
                          window_confidence=window_confidence)
        e_raw = (weights.get("blame_overlap", 0) * b
                 + weights.get("changed_in_window", 0) * x
                 + weights.get("cochange_coupling", 0) * k
                 + weights.get("churn_surprise", 0) * r)
        e_clipped = _clip(e_raw, -CAP_TOTAL_E, CAP_TOTAL_E)
        gate = content_gate(cand.prior)
        adjustment = window_confidence * gate * e_clipped
        score = _logit(cand.prior) + adjustment

        reasons = []
        if b > 0:
            reasons.append("most recent touch decays to blame_overlap=%.2f" % b)
        if x:
            reasons.append("changed within the evaluated window")
        if k > 0:
            reasons.append("co-changes with other window-active files "
                            "(coupling=%.2f)" % k)
        if r > 0.5:
            reasons.append("recent activity is unusual for this file "
                            "(churn_surprise=%.2f)" % r)
        warnings = []
        if gate == 0.0 and e_raw > 0:
            warnings.append("content prior gated the git adjustment to zero "
                             "(churn-trap guard)")
        out.append(RankedSuspect(entity=cand.entity, score=score,
                                  content_score=cand.prior, git_evidence=ev,
                                  reasons=tuple(reasons),
                                  warnings=tuple(warnings)))
    out.sort(key=lambda r: (-r.score, r.entity))
    return out


# ============================================== end-to-end (real git) helper
def rank_suspects_live(root: str, subtree: str, candidates: Sequence[Candidate],
                        anchor: str = "HEAD", window: int = 200,
                        baseline_window: int = 500,
                        **kwargs) -> List[RankedSuspect]:
    """Convenience wrapper doing the real git walk. Kept separate from
    ``rank_suspects`` so every scoring path is testable with injected commit
    lists (no live git in the unit-test suite — rule #190/#218 discipline:
    tests certify the math, not a slow/CPU-contended git walk)."""
    commits = walk_commits_from(root, anchor, subtree, window)
    baseline = walk_commits_from(root, anchor, subtree, baseline_window)
    return rank_suspects(candidates, commits, baseline=baseline, **kwargs)


# ===================================================================== calibration
#: repo-name (as recorded in results/block_templates.json provenance) ->
#: filesystem root. Mirrors probes/block_miner.py's DEFAULT_SOURCES mapping;
#: not imported from there because block_miner hardcodes its own corpus path
#: and importing it would drag a bench-corpus registry into this module.
REPO_ROOT = os.path.dirname(ROOT)


def _repo_map() -> Dict[str, str]:
    return {
        "lab-repo": REPO_ROOT,
        "mission-control": os.path.join(REPO_ROOT, "mission-control"),
    }


def mined_fixes(templates_path: Optional[str] = None) -> List[dict]:
    """Every unique (repo, sha, path) fix-commit provenance record mined by
    probes/block_miner.py, deduplicated. This IS the "~1.9k fix commits"
    corpus's per-file provenance (`/tmp/BURN_LIST.md` correction, line 22):
    the miner walked ~1.9k fix-shaped commits and recorded, per accepted
    added block, exactly which (repo, sha, path, fn) it came from."""
    path = templates_path or os.path.join(ROOT, "results", "block_templates.json")
    with open(path) as fh:
        data = json.load(fh)
    seen = set()
    out = []
    for t in data.get("templates", []):
        for p in t.get("provenance", []):
            key = (p.get("repo"), p.get("sha"), p.get("path"))
            if key in seen or not all(key):
                continue
            seen.add(key)
            out.append(p)
    out.sort(key=lambda p: (p["repo"], p["sha"], p["path"]))
    return out


@dataclass(frozen=True)
class CalibrationCase:
    repo: str
    root: str
    sha: str
    path: str
    subtree: str


def calibration_cases(subtree: str = "backend/app",
                       templates_path: Optional[str] = None,
                       repo_map: Optional[Mapping[str, str]] = None) -> List[CalibrationCase]:
    rmap = dict(repo_map) if repo_map is not None else _repo_map()
    out = []
    for p in mined_fixes(templates_path):
        root = rmap.get(p["repo"])
        if root is None or not is_git_repo(root):
            continue
        out.append(CalibrationCase(repo=p["repo"], root=root, sha=p["sha"],
                                    path=p["path"], subtree=subtree))
    return out


def _candidate_pool_for_case(case: CalibrationCase, window: int
                              ) -> Tuple[List[CommitRecord], List[Candidate]]:
    """Content-blind candidate pool: every file touched in the pre-fix window
    PLUS the true fixed file (so it is always a reachable candidate — the
    doc's §11 ground truth is the fix hunk's file, not a locator's output).
    """
    commits = walk_commits_from(case.root, case.sha + "^", case.subtree, window)
    files = {f for c in commits for f in c.files}
    files.add(case.path)
    candidates = [Candidate(entity=f, content_score=None) for f in sorted(files)]
    return commits, candidates


def calibrate(subtree: str = "backend/app", window: int = 40,
              baseline_window: int = 200, limit: Optional[int] = None,
              templates_path: Optional[str] = None,
              top_k: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """Replay every mined fix against its PRE-FIX snapshot (doc §3, never the
    fix commit itself or anything after it) and measure: with content ignored
    (uniform prior — the churn-trap gate wide open), does the git-evidence
    channel alone rank the true fixed file near the top of every file that
    was actually active in the window?

    Reports the real hit-rate table AND a raw-churn baseline (rank purely by
    touch count) so the churn-trap guard's value is a measured number, not an
    assertion — this is the acceptance condition the doc's §12 names:
    "recent changes improve recall ... without allowing repository hotness
    ... to dominate."
    """
    cases = calibration_cases(subtree=subtree, templates_path=templates_path)
    if limit is not None:
        cases = cases[:limit]

    hits = {k: 0 for k in top_k}
    churn_hits = {k: 0 for k in top_k}
    n_evaluated = 0
    n_skipped_no_window = 0
    ranks: List[int] = []

    for case in cases:
        commits, candidates = _candidate_pool_for_case(case, window)
        if not commits:
            n_skipped_no_window += 1
            continue
        baseline = walk_commits_from(case.root, case.sha + "^", case.subtree,
                                      baseline_window)
        ranked = rank_suspects(candidates, commits, baseline=baseline)
        order = [r.entity for r in ranked]
        if case.path not in order:
            n_skipped_no_window += 1
            continue
        n_evaluated += 1
        pos = order.index(case.path) + 1
        ranks.append(pos)
        for k in top_k:
            if pos <= k:
                hits[k] += 1

        # raw-churn baseline: sort candidates by raw touch count only.
        touch = Counter(f for c in commits for f in c.files)
        churn_order = sorted((c.entity for c in candidates),
                              key=lambda f: (-touch.get(f, 0), f))
        cpos = churn_order.index(case.path) + 1 if case.path in churn_order else None
        for k in top_k:
            if cpos is not None and cpos <= k:
                churn_hits[k] += 1

    def _rate(d):
        return {k: (round(v / n_evaluated, 4) if n_evaluated else None)
                for k, v in d.items()}

    return {
        "cases_total": len(cases),
        "cases_evaluated": n_evaluated,
        "cases_skipped_no_repo_or_window": len(cases) - n_evaluated - n_skipped_no_window
        if len(cases) else 0,
        "cases_skipped_true_file_not_in_window": n_skipped_no_window,
        "window": window,
        "baseline_window": baseline_window,
        "hit_rate_git_evidence": _rate(hits),
        "hit_rate_raw_churn_baseline": _rate(churn_hits),
        "median_rank": (sorted(ranks)[len(ranks) // 2] if ranks else None),
        "note": "content-blind: content_score is a uniform prior for every "
                "candidate (churn-trap gate wide open). This measures the "
                "git-evidence channel alone, never a locate-accuracy claim "
                "(no content locator is wired — see module docstring).",
    }


def _print_report(report: dict) -> None:
    print("BLAME-CHANNEL CALIBRATION — content-blind, real mined fix history")
    print("cases: %d total, %d evaluated, %d had no repo, %d had the true "
          "file outside the pre-fix window"
          % (report["cases_total"], report["cases_evaluated"],
             report["cases_skipped_no_repo_or_window"],
             report["cases_skipped_true_file_not_in_window"]))
    print("window=%d baseline_window=%d" % (report["window"], report["baseline_window"]))
    print()
    print("%-8s %-18s %-18s" % ("top-k", "git_evidence", "raw_churn_baseline"))
    for k in sorted(report["hit_rate_git_evidence"]):
        print("%-8s %-18s %-18s" % (
            k, report["hit_rate_git_evidence"][k],
            report["hit_rate_raw_churn_baseline"][k]))
    print("median_rank:", report["median_rank"])
    print()
    print(report["note"])


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="blame_channel")
    ap.add_argument("command", choices=("calibrate",), nargs="?", default="calibrate")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--baseline-window", type=int, default=200)
    args = ap.parse_args(argv)
    report = calibrate(limit=args.limit, window=args.window,
                        baseline_window=args.baseline_window)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
