"""Line-level localization: code-side pruning, a bounded ask, and a certificate.

WHY THIS MODULE EXISTS (measured, 2026-08-29)
---------------------------------------------
Tasks on the live board now reach the gold FILE and the gold FUNCTION and then
die at `line_emission`. Measured from `results/unified_gym.jsonl` (the six rows
with `run_at >= 1788030000`, all ABSTAIN):

  * 5 of 6 rows carry a `line_emission` stage and EVERY ONE of them is the
    string `"no admissible emission"`.
  * That record is emitted from INSIDE `if lpick is not None:` in
    `fixer.py:6414`, so the line MENU picked a line every time. The stage is
    not starving and it is not running out of candidates.
  * Across the whole 1220-row ledger there is exactly ONE
    `line_emission_rejected` record. So the structural guard
    (`fixer._emission_defect`) is almost never what kills the emission; the
    retries are dying on the three UNRECORDED branches above it — empty output,
    `degenerate(...)`, "same as the old line", and `SyntaxError`.
  * Upstream, `mutation_selector` reached rung 2 by abstaining, with reasons
    `"abstain: no candidate affirmed"` (all probe scores negative) and
    `"abstain: confirm vetoed winner"`.

So the wall is NOT emission starvation. It is that rung 2 asks a 1B model to
FREE-FORM AUTHOR a replacement for one model-chosen line, and then treats that
one authored string as the only anchor in the universe. Two failure classes are
fused into one silent outcome: the model chose the wrong line, and the model
could not write the line. Neither is recorded.

`a_granularity_law.md` predicts exactly this: line-within-function is HYBRID —
the model's nearest alternatives (adjacent lines in the same function) are not
stably separable, so a model choice must never be a recall gate. The design in
`a_line_emission_stage.md` follows: build a code-side plausible universe `P`,
show the model at most a small menu `M`, and let downstream gates arbitrate over
all of `P` regardless of what the model said.

THE INVARIANT THIS MODULE ENFORCES
----------------------------------
    The model may REORDER `P`. The model may never REMOVE from `P`.

Everything else here — budgets, retry policy, certificate shape — exists to
make that invariant provable after the fact.

This module is pure and offline: every model interaction goes through an
injected `LineAsk`, and every filesystem/git interaction is a caller-supplied
argument. It imports nothing from `fixer`.

See `LINE_STAGE_SEAM` for the exact wiring contract.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# SEAM
# --------------------------------------------------------------------------- #

#: The one call `fixer` makes. Mirrors `block_holes.BLOCK_HOLE_SEAM` in shape:
#: a single string naming the contract, so a wiring change is greppable.
LINE_STAGE_SEAM = (
    "line_stage.run_line_stage(file_lines, fn_name, span_lo, span_hi, "
    "symptom='', history_lines=(), ask=None, budget=None, path='auto', "
    "limits=LIMITS) -> LineStageResult(candidates, anchors, cert, state)")

SCHEMA_VERSION = "line-stage/1"

# --------------------------------------------------------------------------- #
# Limits (the design's candidate-count table, §1 "Candidate counts")
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Limits:
    """Bounds on the candidate universe and on what the model may see.

    `p_max` bounds `P` (the enumerated anchor universe). `m_max` bounds `M`
    (what the model sees). `probe_max` is the largest set for which per-
    candidate binary probes are cheaper-than-a-menu AND defensible. Above
    `skip_above` the design says skip the model entirely: this is exactly the
    30-60 anchor case where line discrimination behaves like function-level
    saturation.
    """

    #: MEASURED on the 21 v3 tasks (`results/real_tasks_v3.jsonl`), gold-line-
    #: in-`P` rate by `p_max`:  30 -> 19/21, 60 -> 19/21, 80 -> 20/21,
    #: 120 -> 21/21, 250 -> 21/21. The MEDIAN |P| is 16 at every one of those
    #: settings — only two of the twenty-one functions are large enough for the
    #: cap to bind at all. So raising the cap to 120 buys the last two tasks
    #: and costs the typical task nothing. This deliberately exceeds the
    #: design's 30-60 figure, on the design's own reasoning: "the cost of
    #: omitting the gold line is total repair failure." The number that must
    #: stay small is repair ATTEMPTS, and that is bounded by the gate cascade
    #: and the wave order, not by |P|.
    p_max: int = 120
    #: MEASURED: over the menu-routed rows, gold-in-menu is 10/13 at m_max=8,
    #: 12/13 at 12, and 12/13 at 16. Twelve is where the curve flattens, and it
    #: is the design's own upper bound for what a 1B model should ever see.
    m_max: int = 12
    probe_max: int = 6
    skip_above: int = 16
    context_lines: int = 2
    #: menu-margin below which ONE re-ask is permitted (see `RETRY POLICY`)
    margin_tau: float = 0.15


LIMITS = Limits()

# --------------------------------------------------------------------------- #
# Emission states (§3 — these are NOT collapsed into a generic failure)
# --------------------------------------------------------------------------- #

SELECTED = "selected"
EXPLICIT_ABSTAIN = "explicit_abstain"
SKIPPED_BUDGET = "skipped_budget"
SKIPPED_LARGE_SET = "skipped_large_set"
SKIPPED_DISABLED = "skipped_disabled"
TIMEOUT = "timeout"
INVALID_OUTPUT = "invalid_output"
TRUNCATED_OUTPUT = "truncated_output"
MODEL_UNAVAILABLE = "model_unavailable"
NO_CANDIDATES = "no_candidates"

#: Every state in which the model contributed NOTHING to ordering. Downstream
#: must behave identically for all of them: enumerate `P` by static score.
NON_SELECTING_STATES = frozenset({
    EXPLICIT_ABSTAIN, SKIPPED_BUDGET, SKIPPED_LARGE_SET, SKIPPED_DISABLED,
    TIMEOUT, INVALID_OUTPUT, TRUNCATED_OUTPUT, MODEL_UNAVAILABLE, NO_CANDIDATES,
})

# --------------------------------------------------------------------------- #
# Provenance of the accepted repair (§4 "Downstream arbitration")
# --------------------------------------------------------------------------- #

FROM_MODEL_SELECTED = "model_selected"
FROM_STATIC_PRIORITY = "statically_prioritized"
FROM_MODEL_REJECTED = "model_did_not_select"
FROM_ENUMERATION = "full_enumeration"

# --------------------------------------------------------------------------- #
# Reasons (provenance flags). Union, never intersection (§1).
# --------------------------------------------------------------------------- #

R_SYMPTOM = "symptom_match"
R_BACKWARD = "backward_slice"
R_FORWARD = "forward_slice"
R_CONTROL = "control_dependency"
R_HISTORY = "changed_in_relevant_revision"
R_EXCEPTION = "exception_path"
R_RETURN = "return_or_effect"
R_STRUCTURAL = "structural_expansion"
R_DEGRADED = "degraded_enumeration"
#: A span carrying NO signal that is in `P` only because the whole function fit
#: under `p_max`. See `FUNCTION FLOOR` in `prune_candidates`.
R_FLOOR = "function_floor"

#: Static priority weights. A symptom match is the strongest single signal we
#: can compute without a model, but it is deliberately NOT overwhelming: the
#: design is explicit that "a symptom match should increase priority, not
#: automatically certify causality" — the symptom line is frequently not the
#: causal line.
WEIGHTS: Dict[str, float] = {
    R_SYMPTOM: 3.0,
    R_BACKWARD: 2.0,
    R_FORWARD: 1.0,
    R_CONTROL: 1.5,
    R_HISTORY: 1.25,
    R_EXCEPTION: 1.0,
    R_RETURN: 0.75,
    R_STRUCTURAL: 0.25,
    R_DEGRADED: 0.0,
    R_FLOOR: 0.0,
}

#: Reasons that are NOT evidence — a candidate carrying only these is in `P` for
#: recall insurance, not because anything pointed at it. `choose_path` routes on
#: the count of candidates that carry real evidence, because a menu of 12
#: zero-signal spans is a menu of noise.
_INERT_REASONS = frozenset({R_FLOOR, R_DEGRADED, R_STRUCTURAL})

# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LineCandidate:
    """One plausible STATEMENT SPAN, never a bare physical line number.

    The design is explicit about why (§4): "Do not certify a raw line number by
    itself. Line numbers change after edits." The certifiable identity is the
    normalized source hash plus the AST node kind; the line numbers are how you
    find it in THIS revision.
    """

    cid: str
    start_line: int                 # 1-based, file coordinates, inclusive
    end_line: int                   # 1-based, file coordinates, inclusive
    col_start: int
    col_end: int
    node_kind: str                  # top-level AST node type, or "unknown"
    block_id: str                   # enclosing compound-statement identity
    src: str
    src_hash: str
    ctx_hash: str
    reasons: Tuple[str, ...]
    score: float
    shown_to_model: bool = False

    def covers(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def to_cert(self) -> Dict:
        return {
            "cid": self.cid,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "node_kind": self.node_kind,
            "block_id": self.block_id,
            "src_hash": self.src_hash,
            "ctx_hash": self.ctx_hash,
            "reasons": list(self.reasons),
            "score": round(self.score, 4),
            "shown_to_model": self.shown_to_model,
        }


@dataclass(frozen=True)
class CandidateUniverse:
    """`P` — everything downstream may need to evaluate."""

    candidates: Tuple[LineCandidate, ...]
    degraded: bool = False
    degrade_reason: str = ""
    truncated: bool = False           # p_max clipped the universe
    considered: int = 0               # spans examined before ranking/clipping
    file_hash: str = ""
    fn_span: Tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.candidates)

    def covering(self, line: int) -> Tuple[LineCandidate, ...]:
        return tuple(c for c in self.candidates if c.covers(line))

    @property
    def evidenced(self) -> Tuple[LineCandidate, ...]:
        """Candidates something actually pointed at. Drives menu routing."""
        return tuple(c for c in self.candidates
                     if set(c.reasons) - _INERT_REASONS)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _normalize(text: str) -> str:
    """Whitespace-normalized source, for a hash that survives re-indentation."""
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Symptom tokenization
# --------------------------------------------------------------------------- #

#: Tokens that appear in essentially every line of Python and therefore carry
#: zero discriminative power. Admitting them makes the symptom signal fire on
#: the whole function, which is the same as not having the signal (the fixer's
#: `_LINKAGE_STOP` exists for the identical reason).
_TOKEN_STOP = frozenset("""
if or and in is not none true false for while return def class self def
import from as with try except finally raise pass else elif lambda del
the a an of to on at by it its this that then when what which be been
something going wrong around error value data result item items get set
""".split())

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_NUM_RE = re.compile(r"\b\d+\b")


def symptom_tokens(symptom: str) -> frozenset:
    """Normalized discriminative tokens from the ticket text.

    Splits camelCase and snake_case, keeps numeric literals (a status code or a
    threshold is often the ONLY token the ticket and the gold line share — the
    fixer learned this the hard way with `target_age > 18`), and drops the
    grammar words.
    """
    out = set()
    for w in _WORD_RE.findall(symptom or ""):
        low = w.lower()
        if len(low) >= 2 and low not in _TOKEN_STOP:
            out.add(low)
        for part in _CAMEL_RE.findall(w):
            p = part.lower()
            if len(p) >= 3 and p not in _TOKEN_STOP:
                out.add(p)
        for part in low.split("_"):
            if len(part) >= 3 and part not in _TOKEN_STOP:
                out.add(part)
    for n in _NUM_RE.findall(symptom or ""):
        out.add(n)
    return frozenset(out)


def _line_tokens(text: str) -> frozenset:
    out = set()
    for w in _WORD_RE.findall(text or ""):
        low = w.lower()
        out.add(low)
        for part in _CAMEL_RE.findall(w):
            out.add(part.lower())
        for part in low.split("_"):
            if part:
                out.add(part)
    for n in _NUM_RE.findall(text or ""):
        out.add(n)
    return frozenset(out)


# --------------------------------------------------------------------------- #
# Statement-span extraction
# --------------------------------------------------------------------------- #

_COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
             ast.AsyncWith)
_EXC_NODES = (ast.Try, ast.Raise, ast.ExceptHandler)


@dataclass
class _Span:
    start: int
    end: int
    col_start: int
    col_end: int
    kind: str
    block_id: str
    reasons: set = field(default_factory=set)


def _find_function(tree: ast.AST, fn_name: str, span_lo: int, span_hi: int):
    """The FunctionDef whose name matches and whose body overlaps the span.

    Name match is preferred; when the name is absent or ambiguous the span
    overlap decides. Returns None when nothing plausible is found — the caller
    then degrades rather than guessing.
    """
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lo = node.lineno
        hi = getattr(node, "end_lineno", None) or node.lineno
        if fn_name and node.name != fn_name:
            continue
        if span_lo and span_hi and (hi < span_lo or lo > span_hi):
            continue
        if best is None or (hi - lo) < (best[1] - best[0]):
            best = (lo, hi, node)
    # A NAME THAT DOES NOT RESOLVE IS NOT A LICENCE TO GUESS. Falling back to
    # "whatever function overlaps the span" would silently prune a DIFFERENT
    # function's statements and report a healthy, non-degraded universe — the
    # fail-plausible shape (#181c). Returning None sends the caller down the
    # degraded ladder, which enumerates the span honestly and says so.
    return best[2] if best is not None else None


def _header_end(node: ast.AST) -> int:
    """Last line of a compound statement's HEADER (its test/iter/items).

    A candidate for an `if` must be the CONDITION, not the whole branch: a
    60-line `if` body is not one hypothesis, and enumerating it as one makes the
    span meaningless. So compound statements contribute only their header.
    """
    ends = [node.lineno]
    for attr in ("test", "iter", "items", "type_comment"):
        sub = getattr(node, attr, None)
        for s in (sub if isinstance(sub, list) else [sub]):
            e = getattr(s, "end_lineno", None)
            if isinstance(e, int):
                ends.append(e)
    return max(ends)


def _collect_spans(fn: ast.AST) -> List[_Span]:
    """Every executable statement span inside `fn`, with structural reasons.

    Compound statements contribute their header only; their bodies contribute
    their own statements. That is what makes "a faulty `if` condition and the
    assignment inside its branch" two RELATED candidates rather than one blob,
    which is what the design asks for.
    """
    spans: List[_Span] = []
    seen = set()

    def block_of(stack: List[str]) -> str:
        return "/".join(stack) or "body"

    def walk(node: ast.AST, stack: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt):
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)) and child is not fn:
                # a NESTED def is one candidate (its signature); its body is a
                # different function's problem
                _add(child, child.lineno, child.lineno, stack, set())
                continue
            if isinstance(child, _COMPOUND):
                reasons = {R_CONTROL}
                if isinstance(child, _EXC_NODES):
                    reasons.add(R_EXCEPTION)
                _add(child, child.lineno, _header_end(child), stack, reasons)
                walk(child, stack + [f"{type(child).__name__}@{child.lineno}"])
                for attr in ("handlers", "orelse", "finalbody", "body"):
                    for sub in (getattr(child, attr, None) or []):
                        if isinstance(sub, ast.ExceptHandler):
                            _add(sub, sub.lineno, sub.lineno, stack,
                                 {R_CONTROL, R_EXCEPTION})
                            walk(sub, stack + [f"Except@{sub.lineno}"])
                continue
            reasons = set()
            if isinstance(child, (ast.Return, ast.Yield)):
                reasons.add(R_RETURN)
            if isinstance(child, ast.Raise):
                reasons.add(R_EXCEPTION)
            _add(child, child.lineno,
                 getattr(child, "end_lineno", None) or child.lineno,
                 stack, reasons)

    def _add(node, start, end, stack, reasons):
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        spans.append(_Span(
            start=start, end=end,
            col_start=getattr(node, "col_offset", 0),
            col_end=getattr(node, "end_col_offset", 0) or 0,
            kind=type(node).__name__,
            block_id=block_of(stack),
            reasons=set(reasons),
        ))

    walk(fn, [])
    return sorted(spans, key=lambda s: (s.start, s.end))


# --------------------------------------------------------------------------- #
# Dataflow signal
# --------------------------------------------------------------------------- #


def _dataflow(fn: ast.AST, rel_path: str, qualname: str):
    """Lazy per-function def-use via `graphs.dataflow`, or None.

    The import is INSIDE the function and guarded: line pruning must degrade to
    the symptom + structural signals rather than fail when the dataflow layer is
    unavailable. `analyze_function` is pure and takes an AST node, so no corpus,
    no minter and no store is materialized — this is the lazy per-function path
    the layer was built for, not a repo build.
    """
    try:
        from graphs.dataflow import analyze_function
    except Exception:                                       # pragma: no cover
        return None
    try:
        return analyze_function(fn, f"lines:{rel_path}:{qualname}",
                                rel_path, qualname)
    except Exception:                                       # pragma: no cover
        return None


def _slice_lines(df, seed_lines: frozenset, seed_names: frozenset):
    """(backward, forward) line sets around the seeded observation.

    backward: definitions that REACH a use on a seed line, plus definitions of
              any name the symptom itself names.
    forward:  uses reached by a definition ON a seed line — where the wrong
              value travels next.

    One hop each way, deliberately. A transitive closure over a 200-line
    function returns the function, and a signal that selects everything is not
    a signal.
    """
    backward, forward = set(), set()
    if df is None:
        return backward, forward
    for def_line, use_line, name in df.reaching:
        if use_line in seed_lines or name in seed_names:
            backward.add(def_line)
        if def_line in seed_lines:
            forward.add(use_line)
    for d in df.definitions:
        if d.name in seed_names:
            backward.add(d.line)
    for u in df.uses:
        if u.name in seed_names:
            forward.add(u.line)
    return backward, forward


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #


def prune_candidates(file_lines: Sequence[str],
                     fn_name: str = "",
                     span_lo: int = 0,
                     span_hi: int = 0,
                     symptom: str = "",
                     history_lines: Sequence[int] = (),
                     rel_path: str = "",
                     limits: Limits = LIMITS) -> CandidateUniverse:
    """Build `P` — the code-side plausible universe. No model, no I/O.

    `span_lo`/`span_hi` are 1-based inclusive file coordinates for the located
    function; they are a HINT used to disambiguate a name, never a hard filter
    on candidates (the function's real AST span wins).

    `history_lines` are 1-based file lines the miner reports as changed in a
    relevant revision. Per §1 this is a PRIORITY BOOST and never a filter: "a
    stale line can still contain the defect, and a recently changed line can
    merely expose it."

    Degrades — it never raises. An unparseable file yields every non-blank,
    non-comment line in the span as a candidate with `R_DEGRADED`, and the
    universe is flagged so the certificate can say the analysis was weak.
    """
    lines = list(file_lines or [])
    file_hash = _sha("\n".join(lines))
    tokens = symptom_tokens(symptom)
    hist = {int(h) for h in (history_lines or []) if isinstance(h, int)}

    tree = None
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as exc:
        return _degraded_universe(lines, span_lo, span_hi, tokens, hist,
                                  file_hash, limits,
                                  "file does not parse: %s" % exc.msg)
    fn = _find_function(tree, fn_name, span_lo, span_hi)
    if fn is None:
        return _degraded_universe(lines, span_lo, span_hi, tokens, hist,
                                  file_hash, limits,
                                  "function %r not found in the parsed tree"
                                  % (fn_name or "<span>"))
    fn_lo = fn.lineno
    fn_hi = getattr(fn, "end_lineno", None) or fn.lineno
    spans = _collect_spans(fn)
    if not spans:
        return _degraded_universe(lines, fn_lo, fn_hi, tokens, hist, file_hash,
                                  limits, "function body has no statements")

    # --- symptom seeding -------------------------------------------------- #
    seed_lines = set()
    for sp in spans:
        text = "\n".join(lines[sp.start - 1:sp.end])
        if tokens and (_line_tokens(text) & tokens):
            sp.reasons.add(R_SYMPTOM)
            seed_lines.add(sp.start)
    seed_names = frozenset(
        t for t in tokens
        if any(t == n.lower() for n in _WORD_RE.findall("\n".join(lines[fn_lo - 1:fn_hi]))))

    # --- dataflow --------------------------------------------------------- #
    df = _dataflow(fn, rel_path or "<mem>", fn_name or getattr(fn, "name", "?"))
    backward, forward = _slice_lines(df, frozenset(seed_lines), seed_names)
    df_truncated = bool(getattr(df, "truncated", False))

    for sp in spans:
        rng = range(sp.start, sp.end + 1)
        if any(l in backward for l in rng):
            sp.reasons.add(R_BACKWARD)
        if any(l in forward for l in rng):
            sp.reasons.add(R_FORWARD)
        if any(l in hist for l in rng):
            sp.reasons.add(R_HISTORY)

    # --- structural expansion: control-dependence parents of a hit -------- #
    # A relevant line's governing predicate is itself a candidate (§1). Without
    # this, a wrong `if` whose BODY matched the symptom is unreachable.
    hit_blocks = {sp.block_id for sp in spans if sp.reasons}
    for sp in spans:
        if sp.reasons:
            continue
        if any(b.startswith(sp.block_id) and b != sp.block_id
               for b in hit_blocks) and sp.kind in (
                   "If", "For", "AsyncFor", "While", "Try", "With",
                   "AsyncWith", "ExceptHandler"):
            sp.reasons.add(R_CONTROL)
            sp.reasons.add(R_STRUCTURAL)
        elif R_CONTROL in sp.reasons or R_EXCEPTION in sp.reasons:
            pass

    considered = len(spans)

    # --- FUNCTION FLOOR --------------------------------------------------- #
    # MEASURED on the 21 v3 tasks: `article_publishable_projection` is a
    # 12-line function with six statement spans, the gold change is two of
    # them, and the ticket's obfuscated symptom ("canonical, expose,
    # publisher") shares NOT ONE token with either. Signal-only pruning kept
    # three spans and threw the gold away — a total loss on a function small
    # enough to enumerate for free.
    #
    # So when the whole function fits under `p_max`, every statement span is an
    # anchor. Zero-signal spans carry `R_FLOOR`, score 0, and sort last, so
    # they never displace a real signal in the menu — they exist purely so that
    # a weak symptom cannot cause a recall loss the model could never recover
    # from. This is the design's own rule applied one level up: pruning must
    # not be a hard gate either.
    if len(spans) <= limits.p_max:
        for sp in spans:
            if not sp.reasons:
                sp.reasons.add(R_FLOOR)
        live = spans
    else:
        live = [sp for sp in spans if sp.reasons]
        if not live:
            # Too big to floor AND nothing matched. Take the spans we have and
            # say the analysis was weak rather than returning an empty universe.
            for sp in spans:
                sp.reasons.add(R_DEGRADED)
            live = spans

    cands = _materialize(live, lines, limits)
    ranked = tuple(sorted(cands, key=lambda c: (-c.score, c.start_line, c.cid)))
    truncated = len(ranked) > limits.p_max
    return CandidateUniverse(
        candidates=ranked[:limits.p_max],
        degraded=df is None or df_truncated,
        degrade_reason=("dataflow layer unavailable" if df is None else
                        ("dataflow truncated: function exceeds the statement cap"
                         if df_truncated else "")),
        truncated=truncated,
        considered=considered,
        file_hash=file_hash,
        fn_span=(fn_lo, fn_hi),
    )


def _degraded_universe(lines, lo, hi, tokens, hist, file_hash, limits, why):
    """Fallback ladder rung 5: enumerate every executable-looking line.

    "if parsing or dataflow extraction fails, enumerate all executable statement
    spans in the located function, excluding only whitespace/comments, and
    record the degraded-analysis state."
    """
    lo = max(1, lo or 1)
    hi = min(len(lines), hi or len(lines))
    spans = []
    for i in range(lo, hi + 1):
        text = lines[i - 1] if i - 1 < len(lines) else ""
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        reasons = {R_DEGRADED}
        if tokens and (_line_tokens(text) & tokens):
            reasons.add(R_SYMPTOM)
        if i in hist:
            reasons.add(R_HISTORY)
        spans.append(_Span(start=i, end=i, col_start=len(text) - len(text.lstrip()),
                           col_end=len(text), kind="unknown",
                           block_id="degraded", reasons=reasons))
    cands = _materialize(spans, lines, limits)
    ranked = tuple(sorted(cands, key=lambda c: (-c.score, c.start_line, c.cid)))
    return CandidateUniverse(
        candidates=ranked[:limits.p_max],
        degraded=True,
        degrade_reason=why,
        truncated=len(ranked) > limits.p_max,
        considered=len(spans),
        file_hash=file_hash,
        fn_span=(lo, hi),
    )


def _materialize(spans: List[_Span], lines: Sequence[str],
                 limits: Limits) -> List[LineCandidate]:
    out = []
    for sp in spans:
        src = "\n".join(lines[sp.start - 1:sp.end])
        c0 = max(1, sp.start - limits.context_lines)
        c1 = min(len(lines), sp.end + limits.context_lines)
        ctx = "\n".join(lines[c0 - 1:c1])
        reasons = tuple(sorted(sp.reasons))
        score = sum(WEIGHTS.get(r, 0.0) for r in reasons)
        out.append(LineCandidate(
            cid="L%d_%d" % (sp.start, sp.end),
            start_line=sp.start, end_line=sp.end,
            col_start=sp.col_start, col_end=sp.col_end,
            node_kind=sp.kind, block_id=sp.block_id,
            src=src, src_hash=_sha(_normalize(src)), ctx_hash=_sha(_normalize(ctx)),
            reasons=reasons, score=score,
        ))
    return out


# --------------------------------------------------------------------------- #
# Budget (§3 — emission starvation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LineBudget:
    """Budget isolation for the line stage.

        B_line <= min(line_frac * total, ceiling)   and never touches repair.

    `repair_reserve` is NON-BORROWABLE. If the budget remaining after reserving
    it is below what a line call costs, the stage skips the model and continues
    with `P`. That is the whole point: line reasoning must never consume the
    budget needed to produce or validate the actual fix.

    #11d applies to `answer_tokens`: on this llama.cpp build the server counts
    HIDDEN reasoning against the same budget, so the caller's `budget_for` must
    wrap this number rather than this number being passed as `max_tokens`. We
    keep the visible cap at 32-64 tokens (IDs only, no prose) precisely so the
    reasoning headroom stays affordable.
    """

    total: float = 1.0
    spent: float = 0.0
    repair_reserve: float = 0.5
    line_frac: float = 0.10
    ceiling: float = 0.15
    call_cost: float = 0.02
    answer_tokens: int = 48

    @property
    def line_allowance(self) -> float:
        return min(self.line_frac * self.total, self.ceiling)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    def allows(self, calls: int = 1) -> Tuple[bool, str]:
        need = self.call_cost * max(1, calls)
        if need > self.line_allowance + 1e-9:
            return False, ("line allowance %.4f cannot fund %d call(s) at %.4f"
                           % (self.line_allowance, calls, self.call_cost))
        if self.remaining - need < self.repair_reserve - 1e-9:
            return False, ("repair reserve %.4f would be breached (remaining "
                           "%.4f, need %.4f)"
                           % (self.repair_reserve, self.remaining, need))
        return True, ""

    def charge(self, calls: int = 1) -> "LineBudget":
        return replace(self, spent=self.spent + self.call_cost * max(1, calls))


# --------------------------------------------------------------------------- #
# The ask
# --------------------------------------------------------------------------- #


class LineAsk:
    """Transport seam for the line model. Injected; never constructed here.

    Two methods, both returning a plain dict so a test double is three lines:

      menu(items) -> {"status": "selected"|"abstain", "line_ids": [...],
                      "confidence": "high"|"medium"|"low",
                      "margin": float|None, "raw": str}
      probe(item) -> {"answer": "yes"|"no"|"uncertain", "score": float|None,
                      "raw": str}

    `items` are `(cid, display)` pairs. A transport that times out, is
    unavailable, or cannot parse must RAISE `LineAskTimeout`,
    `LineAskUnavailable`, or return a dict whose `status`/`answer` is
    unrecognized — each maps to its own distinct state. Nothing is collapsed
    into a generic failure.
    """

    def menu(self, items):                                  # pragma: no cover
        raise NotImplementedError

    def probe(self, item):                                  # pragma: no cover
        raise NotImplementedError


class LineAskTimeout(Exception):
    pass


class LineAskUnavailable(Exception):
    pass


class LineAskTruncated(Exception):
    pass


_ID_RE = re.compile(r"L\d+_\d+")


def parse_menu_answer(payload, offered: Sequence[str]):
    """(ids, confidence, error) from a menu payload. IDs only, never prose.

    An id the call did not offer can never survive: the parser intersects with
    `offered`. That is the same containment `fixer._coerce_microcall_answer`
    applies to micro-calls — leniency on our side of the transport, and nothing
    hallucinated gets through.
    """
    offered_set = set(offered)
    if not isinstance(payload, dict):
        return (), "", "payload is not a dict"
    status = str(payload.get("status") or "").strip().lower()
    raw = payload.get("raw")
    if status == "abstain":
        return (), str(payload.get("confidence") or ""), ""
    if status != "selected":
        return (), "", "unrecognized status %r" % status
    ids = payload.get("line_ids")
    if isinstance(ids, str):
        ids = _ID_RE.findall(ids)
    if not isinstance(ids, (list, tuple)):
        if isinstance(raw, str):
            ids = _ID_RE.findall(raw)
        else:
            return (), "", "line_ids is not a list"
    keep = tuple(i for i in dict.fromkeys(str(x).strip() for x in ids)
                 if i in offered_set)
    if not keep:
        return (), "", "no offered id present in the answer"
    conf = str(payload.get("confidence") or "").strip().lower()
    if conf not in ("high", "medium", "low", ""):
        conf = ""
    return keep, conf, ""


def parse_probe_answer(payload):
    if not isinstance(payload, dict):
        return "", None, "payload is not a dict"
    ans = str(payload.get("answer") or "").strip().lower()
    if ans not in ("yes", "no", "uncertain"):
        return "", None, "unrecognized answer %r" % ans
    score = payload.get("score")
    if not isinstance(score, (int, float)):
        score = None
    return ans, score, ""


def display_for(c: LineCandidate, width: int = 90) -> str:
    """One line of menu text: id, line number, reasons, source.

    Reasons go IN the display on purpose. `C` in the granularity law is
    contrastive comparability: two adjacent lines that look identical become
    separable only when the prompt says WHY each is here. Snippets are truncated
    to a common width because option length is a known bias channel on this
    model (the fixer's block-role grammar exists for the same reason).
    """
    head = c.src.strip().splitlines()[0] if c.src.strip() else ""
    why = "+".join(r.replace("_", " ").split()[0] for r in c.reasons
                   if r != R_STRUCTURAL) or "span"
    return "%s (line %d, %s): %s" % (c.cid, c.start_line, why,
                                     head[:width])


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PATH_ENUMERATE = "enumerate"
PATH_MENU = "menu"
PATH_AUTO = "auto"
PATHS = (PATH_ENUMERATE, PATH_MENU, PATH_AUTO)


def choose_path(n: int, path: str, limits: Limits = LIMITS) -> Tuple[str, str]:
    """(effective_path, why). `auto` follows the design's count table."""
    if path == PATH_ENUMERATE:
        return PATH_ENUMERATE, "knob"
    if path == PATH_MENU:
        return (PATH_MENU, "knob") if n else (PATH_ENUMERATE, "empty universe")
    if n == 0:
        return PATH_ENUMERATE, "empty universe"
    if n > limits.skip_above:
        return PATH_ENUMERATE, ("%d candidates exceeds skip_above=%d"
                                % (n, limits.skip_above))
    return PATH_MENU, "%d candidates within the menu band" % n


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class LineStageResult:
    universe: CandidateUniverse
    #: `P`, ordered for downstream evaluation. Model-selected first when the
    #: model selected, then static priority, then the rest. ALWAYS all of `P`.
    anchors: Tuple[LineCandidate, ...]
    state: str
    selected: Tuple[str, ...]
    cert: Dict

    @property
    def candidates(self) -> Tuple[LineCandidate, ...]:
        return self.universe.candidates

    def gold_covered(self, line: int) -> bool:
        return any(c.covers(line) for c in self.anchors)


def _order_anchors(universe: CandidateUniverse,
                   selected: Sequence[str]) -> Tuple[LineCandidate, ...]:
    """selected -> high static score -> the rest. NOTHING is dropped.

    This function is the enforcement point for the module's one invariant, and
    the assertion below is deliberate: a refactor that lets a model selection
    shrink the anchor list is a catastrophic false negative that no test of the
    happy path would notice.
    """
    sel = [c for c in universe.candidates if c.cid in set(selected)]
    rest = [c for c in universe.candidates if c.cid not in set(selected)]
    out = tuple(sel + rest)
    assert len(out) == len(universe.candidates), "anchors must never shrink P"
    return out


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def run_line_stage(file_lines: Sequence[str],
                   fn_name: str = "",
                   span_lo: int = 0,
                   span_hi: int = 0,
                   symptom: str = "",
                   history_lines: Sequence[int] = (),
                   rel_path: str = "",
                   ask: Optional[LineAsk] = None,
                   budget: Optional[LineBudget] = None,
                   path: str = PATH_AUTO,
                   limits: Limits = LIMITS,
                   model_id: str = "",
                   enabled: bool = True) -> LineStageResult:
    """Prune, optionally ask, and return the full anchor universe + a cert.

    RETRY POLICY (the one place the design and the mission brief have to be
    reconciled, so it is written down rather than inferred):

      * A MALFORMED, TRUNCATED, TIMED-OUT or UNAVAILABLE call is NEVER retried.
        §3 is explicit and the reason is budget isolation — a model that cannot
        answer the schema will not answer it on the second try either, and the
        second try is charged to the repair.
      * A call that PARSED CLEANLY but returned a margin below `margin_tau` may
        be re-asked EXACTLY ONCE, and only if the budget still allows two calls
        against the repair reserve. A thin margin is the one case where the
        model did answer and the answer carries no information; re-asking is
        then the cheap way to tell noise from signal.
      * Either way the outcome only REORDERS `P`.
    """
    universe = prune_candidates(file_lines, fn_name, span_lo, span_hi,
                               symptom, history_lines, rel_path, limits)
    cert = _new_cert(universe, fn_name, rel_path, symptom, limits, model_id)
    cert["universe"]["evidenced"] = len(universe.evidenced)
    n = len(universe)
    if n == 0:
        cert["model"]["state"] = NO_CANDIDATES
        return LineStageResult(universe, (), NO_CANDIDATES, (), cert)

    if not enabled:
        cert["model"]["state"] = SKIPPED_DISABLED
        cert["model"]["skip_reason"] = "line emission disabled by config"
        return _finish(universe, (), SKIPPED_DISABLED, cert)

    # Routing is decided by the EVIDENCED count, never by |P|: the function
    # floor deliberately inflates `P` for recall, and letting that inflation
    # push every task onto the enumerate path would trade the model away to
    # pay for insurance we added for free.
    # `or n`: when NOTHING carries evidence the universe is floor-only, which
    # only happens for a function small enough to enumerate whole. Showing that
    # whole small function to the model is the one useful move left, so the
    # count falls back to |P| rather than routing a 6-line function to the
    # skip-the-model path on a technicality.
    n_ev = len(universe.evidenced) or n
    eff_path, why = choose_path(n_ev, path, limits)
    cert["model"]["path"] = eff_path
    cert["model"]["path_reason"] = why
    if eff_path == PATH_ENUMERATE:
        if path == PATH_ENUMERATE:
            state = SKIPPED_DISABLED
            cert["model"]["skip_reason"] = "enumerate path selected by knob"
        else:
            state = SKIPPED_LARGE_SET
            cert["model"]["skip_reason"] = why
        cert["model"]["state"] = state
        return _finish(universe, (), state, cert)

    if ask is None:
        cert["model"]["state"] = MODEL_UNAVAILABLE
        cert["model"]["skip_reason"] = "no LineAsk transport supplied"
        return _finish(universe, (), MODEL_UNAVAILABLE, cert)

    bud = budget or LineBudget()
    use_probes = n_ev <= limits.probe_max
    want_calls = n if use_probes else 1
    ok, reason = bud.allows(want_calls)
    if not ok and use_probes:
        # Probes are the OPTIONAL luxury: a menu is one call and always cheaper.
        # Downgrading beats skipping, so try the menu before giving up.
        use_probes = False
        want_calls = 1
        ok, reason = bud.allows(1)
    if not ok:
        cert["model"]["state"] = SKIPPED_BUDGET
        cert["model"]["skip_reason"] = reason
        cert["model"]["budget"]["blocked"] = True
        return _finish(universe, (), SKIPPED_BUDGET, cert)

    # A zero-signal floor span is never worth a menu slot: it has nothing to
    # say to the model beyond "this line exists". It stays a full anchor.
    shown = universe.candidates[:min(limits.m_max, n_ev)]
    items = [(c.cid, display_for(c)) for c in shown]
    cert["model"]["presented"] = [c.cid for c in shown]
    cert["model"]["menu_hash"] = _sha("|".join(d for _, d in items))
    cert["model"]["mode"] = "probe" if use_probes else "menu"

    if use_probes:
        state, selected, detail = _run_probes(ask, items)
        bud = bud.charge(len(items))
    else:
        state, selected, detail = _run_menu(ask, items, limits)
        calls = 1 + int(bool(detail.get("reasked")))
        bud = bud.charge(calls)
    cert["model"].update(detail)
    cert["model"]["state"] = state
    cert["model"]["selected"] = list(selected)
    cert["model"]["budget"]["spent_after"] = round(bud.spent, 6)
    cert["model"]["budget"]["remaining_after"] = round(bud.remaining, 6)
    return _finish(universe, selected, state, cert)


def _run_probes(ask, items):
    """Binary probes for a very small set. Ordering only; `no` never deletes."""
    detail = {"probe_answers": {}, "probe_scores": {}}
    selected = []
    for cid, display in items:
        try:
            payload = ask.probe((cid, display))
        except LineAskTimeout:
            return TIMEOUT, tuple(selected), detail
        except LineAskUnavailable:
            return MODEL_UNAVAILABLE, tuple(selected), detail
        except LineAskTruncated:
            return TRUNCATED_OUTPUT, tuple(selected), detail
        except Exception as exc:                            # transport blew up
            detail["transport_error"] = type(exc).__name__
            return MODEL_UNAVAILABLE, tuple(selected), detail
        ans, score, err = parse_probe_answer(payload)
        if err:
            detail["probe_answers"][cid] = "invalid"
            continue                      # one bad probe is not a bad stage
        detail["probe_answers"][cid] = ans
        if score is not None:
            detail["probe_scores"][cid] = round(float(score), 4)
        if ans == "yes":
            selected.append(cid)
    if not selected:
        return EXPLICIT_ABSTAIN, (), detail
    return SELECTED, tuple(selected), detail


def _run_menu(ask, items, limits):
    """One constrained menu, plus at most one thin-margin re-ask."""
    offered = [cid for cid, _ in items]
    detail = {"reasked": False}
    try:
        payload = ask.menu(items)
    except LineAskTimeout:
        return TIMEOUT, (), detail
    except LineAskUnavailable:
        return MODEL_UNAVAILABLE, (), detail
    except LineAskTruncated:
        return TRUNCATED_OUTPUT, (), detail
    except Exception as exc:
        detail["transport_error"] = type(exc).__name__
        return MODEL_UNAVAILABLE, (), detail

    ids, conf, err = parse_menu_answer(payload, offered)
    detail["confidence"] = conf
    margin = payload.get("margin") if isinstance(payload, dict) else None
    detail["margin"] = margin
    if err:
        detail["invalid_reason"] = err
        return INVALID_OUTPUT, (), detail
    if not ids:
        return EXPLICIT_ABSTAIN, (), detail

    if isinstance(margin, (int, float)) and margin < limits.margin_tau:
        detail["reasked"] = True
        detail["reask_reason"] = ("margin %.4f below tau %.4f"
                                  % (margin, limits.margin_tau))
        try:
            payload2 = ask.menu(items)
        except Exception as exc:
            # The FIRST answer stands. A failed re-ask never destroys a parsed
            # selection — that would be the model's noise deleting its own
            # signal.
            detail["reask_error"] = type(exc).__name__
            return SELECTED, ids, detail
        ids2, conf2, err2 = parse_menu_answer(payload2, offered)
        detail["reask_ids"] = list(ids2)
        detail["reask_agreed"] = bool(set(ids2) & set(ids))
        if err2 or not ids2:
            return SELECTED, ids, detail
        if not detail["reask_agreed"]:
            # Two disagreeing thin-margin answers ARE the abstain signal. The
            # union would double the model's noise; abstaining costs nothing
            # because every candidate is still an anchor.
            detail["invalid_reason"] = "thin-margin re-ask disagreed"
            return EXPLICIT_ABSTAIN, (), detail
        merged = tuple(dict.fromkeys(list(ids) + list(ids2)))
        detail["confidence"] = conf2 or conf
        return SELECTED, merged, detail
    return SELECTED, ids, detail


def _finish(universe, selected, state, cert):
    anchors = _order_anchors(universe, selected)
    shown = set(cert["model"].get("presented") or [])
    cert["candidates"] = [
        dict(c.to_cert(), shown_to_model=(c.cid in shown))
        for c in universe.candidates]
    cert["anchor_order"] = [c.cid for c in anchors]
    cert["invariant"] = {
        "anchors_equal_universe": len(anchors) == len(universe.candidates),
        "model_removed_candidates": 0,
    }
    return LineStageResult(universe, anchors, state, tuple(selected), cert)


def _new_cert(universe, fn_name, rel_path, symptom, limits, model_id):
    return {
        "schema": SCHEMA_VERSION,
        "source": {
            "rel_path": rel_path,
            "file_hash": universe.file_hash,
            "function": fn_name,
            "fn_start": universe.fn_span[0],
            "fn_end": universe.fn_span[1],
        },
        "universe": {
            "size": len(universe),
            "considered": universe.considered,
            "truncated_at_p_max": universe.truncated,
            "degraded": universe.degraded,
            "degrade_reason": universe.degrade_reason,
            "symptom_tokens": sorted(symptom_tokens(symptom)),
        },
        "limits": {
            "p_max": limits.p_max, "m_max": limits.m_max,
            "probe_max": limits.probe_max, "skip_above": limits.skip_above,
            "margin_tau": limits.margin_tau,
        },
        "model": {
            "model_id": model_id,
            "state": "",
            "path": "",
            "path_reason": "",
            "mode": "",
            "presented": [],
            "menu_hash": "",
            "selected": [],
            "skip_reason": "",
            "budget": {"blocked": False},
        },
        "candidates": [],
        "anchor_order": [],
        "arbitration": [],
        "outcome": {},
    }


# --------------------------------------------------------------------------- #
# Downstream arbitration record (§4)
# --------------------------------------------------------------------------- #


def record_gate(cert: Dict, cid: str, gate: str, passed: bool,
                reason: str = "") -> None:
    """One deterministic gate result for one anchor. Cheap and always recorded.

    The cascade this feeds is the cost control for the 30-60 anchor case: AST
    applicability, then dataflow consistency, then patch-shape feasibility, then
    targeted tests, and only then an expensive repair attempt. 30-60 LOCATION
    HYPOTHESES is affordable; 30-60 full repair attempts is not.
    """
    cert.setdefault("arbitration", []).append({
        "cid": cid, "gate": gate, "passed": bool(passed),
        "reason": reason or "",
    })


def finalize_outcome(cert: Dict, accepted_cid: Optional[str] = None,
                     patch_hash: str = "", validation: str = "") -> Dict:
    """Stamp WHERE the accepted repair came from.

    This is the field that answers the only question that matters about this
    stage: is line emission adding value, or is the static ranking carrying it?
    Without it the stage is unfalsifiable.
    """
    selected = set(cert.get("model", {}).get("selected") or [])
    order = cert.get("anchor_order") or []
    state = cert.get("model", {}).get("state") or ""
    if accepted_cid is None:
        prov = ""
    elif accepted_cid in selected:
        prov = FROM_MODEL_SELECTED
    elif state in NON_SELECTING_STATES:
        # No model opinion existed. Was it the top static anchor or a deep one?
        prov = (FROM_STATIC_PRIORITY if order[:1] == [accepted_cid]
                else FROM_ENUMERATION)
    else:
        prov = FROM_MODEL_REJECTED
    cert["outcome"] = {
        "accepted_cid": accepted_cid,
        "provenance": prov,
        "patch_hash": patch_hash,
        "validation": validation,
        "model_state": state,
    }
    return cert["outcome"]


# --------------------------------------------------------------------------- #
# Offline pre-measurement (the E104 preflight)
# --------------------------------------------------------------------------- #


def gold_lines_from_pair(parent_src: str, fixed_src: str) -> Tuple[int, ...]:
    """Parent-side 1-based lines the gold patch touches.

    A `replace`/`delete` opcode contributes its parent lines directly. An
    `insert` contributes its ANCHOR — the parent line the insertion lands
    against — because a pure insertion has no parent line of its own and an
    anchor is what a line-level localizer can actually be asked to find.
    """
    import difflib
    a = (parent_src or "").splitlines()
    b = (fixed_src or "").splitlines()
    out = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
            None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            out.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            out.add(max(1, min(len(a), i1 + 1)))
            out.add(max(1, min(len(a), i1)))
    return tuple(sorted(x for x in out if x >= 1))


@dataclass
class PreMeasureRow:
    key: str
    fn: str
    path: str
    family: str
    gold_lines: Tuple[int, ...]
    universe_size: int
    shown_size: int
    gold_in_universe: bool
    gold_in_shown: bool
    gold_rank: Optional[int]
    degraded: bool
    effective_path: str


def premeasure_task(task: Dict, path: str = PATH_AUTO,
                    limits: Limits = LIMITS) -> PreMeasureRow:
    """One v3 row -> gold-line recall over the code-side universe. No model.

    This is the E104 preflight the granularity law asks for: measure whether the
    GOLD LINE survives pruning BEFORE deciding what to ask the model, because
    the model can only ever reorder what pruning left behind.
    """
    parent = task.get("parent_src") or ""
    lines = parent.splitlines()
    fn = task.get("fn") or ""
    gold = gold_lines_from_pair(parent, task.get("fixed_src") or "")
    uni = prune_candidates(lines, fn, 0, 0, task.get("symptom") or "",
                           (), task.get("rel") or "", limits)
    n_ev = len(uni.evidenced) or len(uni)
    eff, _ = choose_path(n_ev, path, limits)
    shown = (uni.candidates[:min(limits.m_max, n_ev)]
             if eff == PATH_MENU else ())
    in_uni = any(c.covers(g) for c in uni.candidates for g in gold)
    in_shown = any(c.covers(g) for c in shown for g in gold)
    rank = None
    for i, c in enumerate(uni.candidates):
        if any(c.covers(g) for g in gold):
            rank = i + 1
            break
    return PreMeasureRow(
        key=str(task.get("sha") or "")[:8], fn=fn,
        path=task.get("rel") or task.get("path") or "",
        family=task.get("family") or task.get("shape") or "",
        gold_lines=gold, universe_size=len(uni), shown_size=len(shown),
        gold_in_universe=in_uni, gold_in_shown=in_shown, gold_rank=rank,
        degraded=uni.degraded, effective_path=eff)


def premeasure(tasks: Sequence[Dict], path: str = PATH_AUTO,
               limits: Limits = LIMITS) -> List[PreMeasureRow]:
    return [premeasure_task(t, path, limits) for t in tasks]


def premeasure_table(rows: Sequence[PreMeasureRow]) -> str:
    n = len(rows) or 1
    hit_u = sum(1 for r in rows if r.gold_in_universe)
    hit_m = sum(1 for r in rows if r.gold_in_shown)
    menu_rows = [r for r in rows if r.effective_path == PATH_MENU]
    sizes = sorted(r.universe_size for r in rows)
    med = sizes[len(sizes) // 2] if sizes else 0
    out = [
        "n=%d" % len(rows),
        "gold-in-universe: %d/%d (%.0f%%)" % (hit_u, len(rows), 100.0 * hit_u / n),
        "gold-in-model-menu: %d/%d (%.0f%%)" % (hit_m, len(rows), 100.0 * hit_m / n),
        "median |P|: %d   max |P|: %d" % (med, sizes[-1] if sizes else 0),
        "rows routed to menu: %d   to enumerate: %d"
        % (len(menu_rows), len(rows) - len(menu_rows)),
        "degraded analyses: %d" % sum(1 for r in rows if r.degraded),
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# WIRING (see LINE_STAGE_WIRING for the exact fixer lines)
# --------------------------------------------------------------------------- #

LINE_STAGE_WIRING = """\
Wiring pass for fixer.py rung 2 (`if new_src is None and cfg.fix.line_fix_fallback:`
at fixer.py:6414). FROZEN until the live board completes; these are the lines.

REPLACE the body of that block. Keep everything before and after untouched.

    import line_stage as _ls

    class _FixerLineAsk(_ls.LineAsk):
        \"\"\"fixer's transport for the line menu. IDs only, 48 answer tokens.\"\"\"
        def menu(self, items):
            menu_text = "\\n".join(d for _cid, d in items)
            text, raw = _call(
                LINE_ID_MENU_Q.format(task=fix_task, fn=fnpick, menu=menu_text),
                seed=cfg.seed, url=cfg.url,
                max_tokens=budget_for(48),          # #11d: budget_for adds the
                n_probs=2, label="line_stage:menu") # hidden-reasoning headroom
            ids = _ls._ID_RE.findall(text or "")
            return {"status": "selected" if ids else "abstain",
                    "line_ids": ids, "confidence": "", "raw": text,
                    "margin": _margin_from(raw)}
        def probe(self, item):
            cid, display = item
            text, raw = _call(LINE_PROBE_Q.format(task=fix_task, cand=display),
                              seed=cfg.seed, url=cfg.url,
                              max_tokens=budget_for(8), n_probs=2,
                              label="line_stage:probe")
            low = (text or "").strip().lower()
            ans = "yes" if low.startswith("yes") else (
                  "no" if low.startswith("no") else "uncertain")
            return {"answer": ans, "score": _margin_from(raw), "raw": text}

    _res = _ls.run_line_stage(
        lines, fnpick, lo + 1, hi,
        symptom=fix_task,
        history_lines=_history_lines_for(fpick),   # miner git idioms; () is fine
        rel_path=fpick,
        ask=_FixerLineAsk(),
        budget=_ls.LineBudget(total=1.0, spent=_budget_spent_frac(),
                              repair_reserve=0.5, call_cost=0.02),
        path=getattr(cfg.fix, "line_stage_path", "auto"),
        model_id=getattr(cfg, "model", ""),
        enabled=cfg.fix.line_fix_fallback)
    cert["stages"].append({"line_stage": _res.cert})

    for _cand in _res.anchors:                     # ALL of P, model-first order
        _ls.record_gate(cert["stages"][-1]["line_stage"], _cand.cid,
                        "ast_applicable", _cand.node_kind != "unknown")
        ...existing per-line emission attempt against _cand, unchanged...
        if new_src is not None:
            _ls.finalize_outcome(cert["stages"][-1]["line_stage"],
                                 _cand.cid, patch_hash=_sha_of(new_src),
                                 validation="compile:pass")
            break
    if new_src is None:
        _ls.finalize_outcome(cert["stages"][-1]["line_stage"], None)
        cert["stages"].append({"line_emission": "no admissible emission",
                               "anchors_tried": len(_res.anchors),
                               "line_state": _res.state})

TWO CONFIG LINES in gym_config.py next to `line_fix_fallback`:

    line_stage_path: str = "auto"   # "auto" | "menu" | "enumerate"
    line_stage_p_max: int = 60

THE MEASUREMENT GAP THIS CLOSES: today's `"no admissible emission"` is a single
opaque string covering four distinct branches. After wiring, the cert names the
model state, every anchor tried, and the gate that rejected each one.
"""
