"""FIXER — the unified task executor. Hand it a task, get a verified fix or an
honest ABSTAIN. The productized composition of everything proven 08-28:

  task text + repo root
    -> C0 bug-class inference (distributional: run top families when unsure)
    -> g1 locate cascade (BM25 + import-graph + 1B relevance atoms -> evidence-card
       file menu)
    -> function menu
    -> FIX LADDER: mutation-menu (code-enumerated candidates, model picks)
                   -> single-line region fix -> atom-assembly (guards/branch)
    -> GATES: compile + syntax + (regression asserts if a test oracle is supplied
       or discoverable) + certificate
    -> output: unified diff + certificate {stage evidence, margins, gates passed}
       or ABSTAIN {reason, stage reached, candidates considered}

Every behavior is knob-routed through `gym_config.GymConfig` (see gym_config.py):
locate/fix/gates/facts/kernel knobs are AUTHORITATIVE — nothing is hardcoded that
the registry names. The certificate carries the config fingerprint + full dict so
an ablation is queryable from the jsonl alone.

Menu decisions request n_probs=8 and record the top-2 logprob margin on the answer
letter (the detreason/kernel.py fragility detector). With cfg.kernel.margin_probe
on, a margin below cfg.kernel.margin_tau re-asks the menu with the candidate order
REVERSED; a disagreement between the two orderings is an ABSTAIN for that menu.

No gold answers anywhere in the loop — this runs on tasks whose answer nobody knows.
Usage:  python3 fixer.py <repo_root> "task description"  [--apply]
Default is dry-run (prints the diff); --apply writes the file.
"""
import ast, difflib, hashlib, http.client, json, math, os, pathlib, re, sys, tempfile, threading, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from swarm.client2 import Reply, degenerate
from probes.real_repo_baseline_v2 import (
    build_index, words, bm25_rank, REL_Q, FILE_MENU_Q, FUNC_MENU_Q)
# NOTE: enumerate_mutations is being refactored into local-swarm-poc/mutations.py
# with a re-export kept here — import from v3 so both orderings work.
from probes.real_repo_baseline_v3 import enumerate_mutations, FILE_MENU_EV_Q, MUT_MENU_Q
import gym_config
import edit_sets as _edit_sets
import prompt_ledger
import function_relevance as _function_relevance

URL = gym_config.GymConfig.url          # legacy module constant (default only)
GIT_ROOT = os.environ.get("STATEFRAME_GIT_ROOT")

# A format exemplar is not an answer. Imported menu templates are shared with
# the probes, so normalize their live first-option exemplar at this boundary
# instead of mutating those probe modules.
def _neutralize_menu_example(template):
    live_first_letter = chr(65)
    return template.replace(
        '{{"pick": "%s"}}' % live_first_letter,
        '{{"pick": "<letter>"}}  (use one displayed menu letter)')


FILE_MENU_EV_Q = _neutralize_menu_example(FILE_MENU_EV_Q)
FILE_MENU_Q = _neutralize_menu_example(FILE_MENU_Q)
FUNC_MENU_Q = _neutralize_menu_example(FUNC_MENU_Q)
MUT_MENU_Q = _neutralize_menu_example(MUT_MENU_Q)

# The imported baseline relevance question is also a model-facing contract. Keep
# its answer example symbolic and make the fixer use a strict field parser below;
# a rationale containing a live word must never become a relevance decision.
REL_Q = """TASK: {task}
Candidate file: {path}
Its top identifiers: {idents}
Could this file plausibly contain the buggy function? Reply with only:
{{"answer": "<yes|no|unsure>"}}  (replace the placeholder)"""


CLASS_Q = """TASK: {task}
What KIND of bug does this describe? Menu:
  A) a comparison/threshold applied in the wrong direction
  B) conditions combined incorrectly (and/or logic)
  C) a None/missing value handled backwards
  D) arithmetic producing wrong results
  E) a step that should happen is not happening
  F) the routine reaches for the wrong dependency/import
  G) a fixed value (constant, key, string) is not the right one
  H) a failure is caught or re-raised in the wrong way
  I) an attribute or type mismatch: the object is right but its property/type is wrong
  J) something else / can't tell
 Reply with only: {{"pick": "<letter>"}}  (use one displayed menu letter)"""

# MEASURED 2026-08-29. The A-E menu could only ever
# name one of the four legacy classes, and a named legacy class ROUTES THE FIX
# STAGE INTO A SINGLE FAMILY (`if cls in enabled_cls`). On the fresh-21 real
# corpus, whose census shapes are import_change / statement_insert /
# literal_change / exception_handler_change / multi_line_coordinated, that
# routing is always wrong: it forces bool_flip-style enumeration and the
# EXPRESSIVE_FAMILIES (literal_substitution, none_passthrough_guard,
# clause_insertion) are never enumerated at all. The census shapes are now
# NAMEABLE, and every one of them maps to a token deliberately OUTSIDE
# FAMILY_MAP so the union path is what runs — see COMPLEX_CLASSES.
CLASS_MAP = {"A": "cmp_flip", "B": "bool_flip", "C": "none_flip",
             "D": "arith_flip", "E": "statement_insert",
             "F": "import_change", "G": "literal_change",
             "H": "exception_handler_change", "I": "attribute_swap", "J": None}

# The census shapes. NON-NEGOTIABLE INVARIANT: no member may appear in
# FAMILY_MAP. A class token that names a family routes the fix stage to that
# ONE enumerator; these shapes have no single enumerator that expresses them, so
# they must fall through to the UNION of every enabled family. The invariant is
# asserted in test_fixer_offline.py rather than left as a comment.
COMPLEX_CLASSES = frozenset((
    "import_change", "statement_insert", "exception_handler_change",
    "literal_change", "multi_line_coordinated", "attribute_swap",
    "guard_insertion",
))

# C0 SHAPE TAXONOMY. Same evidence class as real_repo_baseline_v3.CLASS_HINTS —
# a deterministic symptom-wording -> class map — for the shapes that corpus's
# hints predate. Checked BEFORE the legacy hints because these are whole-clause
# phrases and the legacy hints are single words: "…around article, boundary and
# tenant: this routine reaches for the wrong dependency" matched the legacy
# `cmp_flip` hint "boundary" on a token from the SYMPTOM NOUN LIST, which is the
# one part of the ticket that says nothing about the bug's kind (measured: task
# 2dbb4371, an import_change, was routed to cmp_flip and enumerated comparisons).
SHAPE_CLASS_HINTS = {
    "attribute_swap": ["wrong property of the right object"],
    "exception_handler_change": ["caught or re-raised in the wrong way"],
    "multi_line_coordinated": ["spans several steps of this routine"],
    "statement_insert": ["a step that should happen is not happening"],
    "import_change": ["reaches for the wrong dependency"],
    "literal_change": ["a fixed value used here is not the one"],
    "guard_insertion": ["is not rejected and is processed anyway"],
}


def infer_symptom_class(task_text):
    """The C0 class for a ticket: census SHAPE first, then the legacy taxonomy.

    Returns a class token or None. Deterministic, no model call.
    """
    tl = (task_text or "").lower()
    for cls, hints in SHAPE_CLASS_HINTS.items():
        if any(h in tl for h in hints):
            return cls
    from probes.real_repo_baseline_v3 import infer_class
    return infer_class(task_text)


def _block_hole_shape_allowed(shape, cfg):
    """Return whether the inferred shape is enabled for block authoring.

    Block insertion/replacement is intentionally opt-in by repair shape: an
    unknown class must not spend model calls, and a malformed override must
    fail closed rather than widening the tranche accidentally.
    """
    if shape is None:
        return False
    fix_cfg = getattr(cfg, "fix", None)
    if fix_cfg is None:
        return False
    allowed = getattr(fix_cfg, "block_hole_shapes",
                      gym_config.BLOCK_HOLE_SHAPE_ALLOWLIST)
    if isinstance(allowed, str):
        allowed = (allowed,)
    try:
        return shape in allowed
    except TypeError:
        return False

# family knob name -> the class token enumerate_mutations routes on.
# ORDER MATTERS: the union path below enumerates in cfg.fix.families order and
# then truncates at menu_cap, so the four measured families stay FIRST and the
# contact families (added 2026-08-28) append.
# CLASS_MAP can never name a contact family — the 1B's A-E class menu only covers
# the four legacy ones — so contact candidates reach the menu exclusively through
# the union path, which is why they must be enabled in cfg.fix.families to exist
# at all (registry entry without a grant path = a dead family).
# NOT IN HERE, DELIBERATELY: `typed_holes` and `block_hole`. FAMILY_MAP is the
# router for `enumerate_mutations`, and both of those families are TRANCHES with
# a model seam rather than enumerators — they have no class token to route on,
# and an entry here would send them into mutations.py, which has never heard of
# them. Their grant path is their own `FixCfg` knob (which a run's config
# fingerprint records), so #181's "registry entry without a grant path is a dead
# family" is satisfied by a different, equally explicit channel. Adding either
# one here would be a silent no-op family, which is the failure #181 is about.
FAMILY_MAP = {"cmp": "cmp_flip", "bool": "bool_flip",
              "none": "none_flip", "arith": "arith_flip",
              "wrong_variable_used": "wrong_variable_used",
              "missing_kwarg": "missing_kwarg",
              "missing_await": "missing_await",
              "boolean_operator": "boolean_operator",
              "early_return_missing": "early_return_missing",
              # mutations.py's attribute family. NOT the same thing as the
              # `attr` knob below it, which swaps against a task-named CLASS's
              # own source; this one swaps against attributes the FILE already
              # uses. A family with no FAMILY_MAP entry has no grant path and
              # can never be enumerated (#181), so the entry and the gym_config
              # knob land in the same change.
              "wrong_attribute": "wrong_attribute",
              # EXPRESSIVE FAMILIES (real-19 autopsy fix #1). A registry entry
              # without a grant path is a dead family (#181), so these land in
              # the same change as their gym_config knob.
              "literal_substitution": "literal_substitution",
              "none_passthrough_guard": "none_passthrough_guard",
              "clause_insertion": "clause_insertion",
              # COORDINATED FAMILY (2026-08-29, edit_sets.py).
              # Its candidates are `EditSet`s: subclasses of the same
              # (line, old, new) tuple carrying an extra `line_span`, so every
              # display/filter/rank path below is unchanged and only
              # `edit_sets.apply_candidate` needs to know the difference.
              # This entry IS the grant path — the 1B's A-E class menu can never
              # name this family, so without it the family is dead (#181).
              "coordinated_edit_set": "coordinated_edit_set"}

# SHAPE ROUTING (2026-08-29). A census SHAPE is not an enumerator — that is the
# COMPLEX_CLASSES invariant and it stands — but some shapes name exactly ONE
# family that can express them, and the union arm has no way to know that. The
# round-robin already fixed family-ORDER bias; what it cannot fix is that a
# 12-family round-robin puts each family's best candidate at index ~12, so a gold
# at rank 2 INSIDE its family lands at index 9 — one slot outside an 8-wide menu.
#
# The route is a PREFERENCE, never a filter. The shaped family's candidates go
# first and the full union is appended behind them, deduped, so no candidate the
# flat union would have offered is lost and a wrong shape inference costs
# ordering, not reachability. That is the fail-safe the "if the shaped menu
# produces no verdict, fall back to the union" requirement asks for, expressed
# structurally instead of as a retry: the fallback is always already in the menu.
#
# Entries must map a shape to EXACTLY ONE family. Deliberately not routed:
# `statement_insert` / `guard_insertion` (clause_insertion AND
# none_passthrough_guard both express them), `literal_change` (literal_substitution
# is the obvious route but nothing has measured it, and an unmeasured route here
# is a claim, not a finding), `multi_line_coordinated` (its family is appended
# after the per-function cap by construction — see edit_sets.FIXER_SEAM).
SHAPE_FAMILY_ROUTES = {
    # ea639fc1 `msg.message` -> `msg.content`: census shape attribute_swap, and
    # `wrong_attribute` is the only family that swaps an attribute against the
    # ones the FILE already uses. Measured (test_fixer_offline.py::
    # test_shape_routing_lands_the_ea639fc1_gold_inside_the_eight_wide_menu):
    # rank 10 of 61 -> rank 2 of 61, which is the gold's rank inside that
    # family and therefore the floor for any router that does not also
    # re-order the family itself.
    "attribute_swap": "wrong_attribute",
    # CENSUS TRANCHE ROUTES (2026-08-29). `import_change` and
    # `exception_handler_change` are ONE-TO-ONE with `coordinated_edit_set`:
    # both shapes are multi-line INSERTIONS (an import statement added to a
    # function body; a `try:` wrapping a run of statements), and every
    # single-line family emits exactly one `(line, old, new)` tuple, so no other
    # enabled family can express either. Measured, not assumed — the route was
    # checked against the whole family registry in
    # test_census_families_offline.py::test_routed_shapes_are_one_to_one.
    "import_change": "coordinated_edit_set",
    "exception_handler_change": "coordinated_edit_set",
    # DELIBERATELY NOT ROUTED — `statement_insert`. The prior lane's note said
    # it had two candidate families; the count is now THREE and the route is
    # therefore further from one-to-one, not closer:
    #   * `clause_insertion` adds a clause to a condition or filter chain;
    #   * `none_passthrough_guard` inserts `if x is None: <action>`;
    #   * `coordinated_edit_set`'s `c2_statement_insert` inserts a commit /
    #     rollback / release / binding / peer step.
    # All three genuinely express "a step that should happen is not happening",
    # so naming one would be a preference dressed as a finding, and the shaped
    # arm puts the chosen family's WHOLE output ahead of the round-robin — i.e.
    # a wrong pick here costs the other two families their interleaved head
    # slots. The round-robin union is the honest arm for this shape until
    # something MEASURES which family carries its golds.
}


def build_mutation_menu(span, lo, cls, lines, fam_enabled, providers=None):
    """The rung-1 candidate pool for one function, in menu order.

    Three arms, most specific first:

    1. the class NAMES an enabled family -> that family alone (unchanged);
    2. the class names a SHAPE this file routes -> that family's candidates
       first, then the whole union appended behind them (nothing is lost);
    3. otherwise -> the union of every enabled family, ROUND-ROBIN.

    Round-robin (arm 3) is the 2026-08-29 fix for a family-order bias wearing a
    union costume: concatenating in `cfg.fix.families` order let the first big
    family eat the head of the pool, so every later family's BEST candidate
    ranked below every earlier family's WORST. Measured on ea639fc1,
    `wrong_variable_used` contributes 30 candidates and the gold — rank 2 of 8
    inside its OWN family — landed at index 42 of 61; round-robin moves it to 9.
    WITHIN a family the existing order is untouched, and a family that produces
    nothing costs nothing.
    """
    enabled_cls = {FAMILY_MAP[f] for f in fam_enabled}
    # A COMPLEX class names a SHAPE, not an enumerator (COMPLEX_CLASSES is
    # disjoint from FAMILY_MAP by invariant), so it can never take arm 1. The
    # condition says so out loud rather than relying on the set difference — a
    # future FAMILY_MAP entry colliding with a shape name would otherwise
    # silently re-route these tasks into one family.
    if cls in enabled_cls and cls not in COMPLEX_CLASSES:
        return list(enumerate_mutations(span, lo, cls, file_lines=lines,
                                        providers=providers))
    per_family = [enumerate_mutations(span, lo, FAMILY_MAP[f], file_lines=lines,
                                      providers=providers)
                  for f in fam_enabled]
    muts, seen, rank = [], set(), 0

    def _take(mut):
        if (mut[0], mut[2]) in seen:
            return
        seen.add((mut[0], mut[2]))
        muts.append(mut)

    shaped = SHAPE_FAMILY_ROUTES.get(cls)
    if shaped is not None and shaped in fam_enabled:
        for mut in per_family[fam_enabled.index(shaped)]:
            _take(mut)
    while True:
        progressed = False
        for cands in per_family:
            if rank >= len(cands):
                continue
            progressed = True
            _take(cands[rank])
        if not progressed:
            break
        rank += 1
    return muts


#: EXPLORATION ceiling for the mutation menu, distinct from `cfg.fix.menu_cap`
#: (the MODEL-facing menu width). Candidates are ranked over this wider pool and
#: only then cut to menu_cap — see the ordering-defect note at the union arm.
MENU_EXPLORE_CAP = 96


# USAGE-FREQUENCY DOMINANCE PRIOR: BUILT, MEASURED DEAD, NOT SHIPPED.
#
# 2026-08-29. The proposal was to promote an attribute swap whose target is the
# DOMINANT attribute on the same receiver in the file's usage corpus, to rescue
# task ea639fc1 (`msg.message` -> `msg.content`) whose AST-gold is enumerable but
# sat at index 42 of 61 on the union path with every task-symbol evidence channel
# scoring it zero (the scrubbed symptom never says "content").
#
# MEASURED on that exact task: the receiver `msg` carries `{replace: 1,
# message: 1}` across the whole parent file. `.content` is read ZERO times on
# it — of course it is, its absence IS the bug. The prior scored 0 for all 61
# candidates, and a scoring channel that never fires is worse than no channel
# (#155): it looks like evidence in the certificate and moves nothing.
#
# What actually carries the signal is already in `mutations._rank_attributes`
# tier 0 — the assignment target is literally named `content` — which is why the
# gold sits at rank 2 of 8 INSIDE its own family. The loss is entirely the union
# arm's family concatenation burying `wrong_attribute` behind
# `wrong_variable_used`'s 30 candidates; the round-robin below moves the same
# gold from index 42 to index 9.
#
# Landing it inside menu_cap=8 needed SHAPE ROUTING, which LANDED 2026-08-29 as
# `SHAPE_FAMILY_ROUTES` + `build_mutation_menu` below. The classifier the route
# needs already existed (`infer_symptom_class` returns "attribute_swap" for this
# task's own scrubbed symptom — measured, not assumed), so the route is a
# preference over the union rather than new inference. Measured on this task:
# rank 10 of 61 -> rank 2 of 61, i.e. inside an 8-wide menu.
# The dead usage-frequency prior above stays dead.


def _cert_change(old, new):
    """One-line 'old -> new' for the certificate.

    A coordinated `EditSet`'s old/new are multi-line BLOCKS, so they are
    flattened the same way the model-facing menu flattens them — a raw newline
    inside a cert `change` string is unreadable in every consumer we have.
    """
    def _f(text):
        return " / ".join(l.strip() for l in str(text).strip().split("\n") if l.strip())
    return f"{_f(old)} -> {_f(new)}"


def _cert_line_span(candidate, fallback_line):
    """Return a JSON-safe 1-based span for any emitted candidate."""
    span = getattr(candidate, "line_span", None)
    if isinstance(span, (list, tuple)) and len(span) == 2:
        start, end = span
        if (isinstance(start, int) and isinstance(end, int)
                and 0 <= start <= end):
            return [start + 1, end + 1]
    return [fallback_line, fallback_line]

# The 1B locate decision is calibrated for the eight-choice menu contract.
# Recall candidates may widen the internal pool, but must never widen the
# model-facing file menu past that contract.
MAX_MODEL_FILE_MENU_OPTIONS = 8
MAX_FILE_RANK_WINDOW = 128


def _file_rank_window(cfg, model_width):
    """Return a bounded pre-cut file-ranking window.

    The model width remains the hard prompt budget; this wider window only lets
    the deterministic ranker promote a candidate before that cut. Invalid or
    zero configuration falls back to the model width, and the hard ceiling
    prevents a config mistake from turning retrieval evidence into an oversized
    prompt or an unbounded ranking pass.
    """
    try:
        width = max(1, int(model_width))
    except (TypeError, ValueError):
        width = MAX_MODEL_FILE_MENU_OPTIONS
    if not getattr(cfg.locate, "file_rank_fusion", False):
        return width
    try:
        configured = int(getattr(cfg.locate, "file_rank_window", 0) or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        return width
    return min(MAX_FILE_RANK_WINDOW, max(width, configured))


def _record_retrieval_miss(cert, stage, target, options, reason):
    """Record a known containment failure separately from model outcomes.

    The target is only supplied by callers that have an oracle or an explicit
    plan target; ordinary model runs never infer a gold path from task prose.
    """
    record = {
        "category": "RETRIEVAL MISS",
        "stage": str(stage),
        "target": target,
        "options": list(options or []),
        "reason": str(reason),
    }
    cert.setdefault("retrieval_misses", []).append(record)
    cert.setdefault("stages", []).append({"retrieval_miss": record})
    return record

# ---------------------------------------------------------------------------
# FILE-MENU RANK FUSION (E51)
#
# The E49 doctrine one stage upstream. Measured over the 21-task gym (3 cycles,
# per-task keyed): the 1B answered option A in 15 of 16 file menus. Where gold
# sat at index 0 it was picked 8/10; where gold sat anywhere else it was picked
# 0/6. The file menu is not a decision either — it is a position readout, so
# FILE-MENU ACCURACY IS RANKER TOP-1 ACCURACY and the lever is ordering.
#
# The autopsy named two concrete ranking defects, both visible in the evidence
# cards the pipeline already prints:
#
#   1. NO LENGTH NORMALIZATION. func_index._bm25 is a k1-only BM25 with no
#      b/avgdl term, so raw term frequency decides and the largest module in the
#      corpus wins mechanically. In all six wrong-at-zero menus the winner was
#      api/lessons.py scoring through its <module> pseudo-record — a record that
#      cannot even hold the bug, since the bug lives in a def.
#   2. TICKET BOILERPLATE OUTVOTING SYMPTOM WORDS. The losers' matched
#      identifiers were {and, bug, fix, the} and {find, function, for, check,
#      direction, wrong} — the ticket TEMPLATE. The gold files matched on
#      {files, number, submission, validate} and {band, grade, given, return,
#      string}. Exactly the rank_query_facts defect (E42) with a different
#      source of noise words.
#
# CONTRACT — this is a STRICT PERMUTATION. It is handed the candidate list the
# old code produced (anchor prefix and truncation already applied) and returns
# the same paths in a different order. It cannot add, drop, widen or narrow
# anything, so menu recall and the primary/overflow split are provably
# unchanged and the worst case is neutral. Guarded by
# test_fixer_offline.py::TestFileRankFusion.
# ---------------------------------------------------------------------------

# Words a bug ticket always contains. They carry no locating signal, but a big
# enough module bag contains every one of them, which is how they came to decide
# the menu. Deliberately generic English + ticket scaffolding ONLY: never a
# domain word, or the ranker would go blind to the symptom it is looking for.
FILE_RANK_BOILERPLATE = frozenset("""
the a an and or of to for in on at by is are be was were it its this that these
those with from as if else not but so than then there here what which who whom
find fix bug bugs function functions method responsible should shouldnt would
could must check checks direction wrong incorrect correct case cases pass passes
fail fails expected value values return returns given code file files test tests
error errors missing none get gets set sets new create creates update updates
make makes when where why how does do doing done use uses used call calls line
lines change changes changed fixed broken issue problem behaviour behavior
""".split())


# ---------------------------------------------------------------------------
# DUNDER EXPANSION (rung-4000 autopsy, ranked fix #6b)
#
# BUILT, MEASURED, AND SHIPPED OFF — see gym_config.LocateCfg.dunder_expansion
# for the numbers. Two findings, both worth keeping:
#   (1) `models/` is NOT excluded from the corpus. load_repo() collects 35
#       models/*.py out of 210 files and `models/learner_concept_gap.py` is among
#       them, so the two-cycle C7 miss is a pure RANKING hole, not a directory
#       absence. That was the autopsy's open question and it is now closed.
#   (2) Expansion does not fix the ranking hole. Against the live corpus the gold
#       file sits at fused rank 45 with the raw query, 45 with any dunder-spelling
#       expansion (the tokenizer splits underscores and drops <3-char tokens, so
#       `__repr__` reduces to `repr`, which the query already had), and 117 — much
#       worse — with English synonyms, which flood the ranking toward prose-heavy
#       files. The knob defaults OFF on that evidence.
# Expansion is RETRIEVAL-ONLY and append-only: the original text is preserved
# verbatim and no prompt the model reads is altered.
DUNDER_LEXICAL = {
    "repr": "repr representation string display format debug",
    "str": "str string text representation display format",
    "init": "init initialize constructor setup create",
    "eq": "eq equal equality compare comparison",
    "lt": "lt less than ordering compare sort",
    "hash": "hash hashable key dict set",
    "len": "len length size count empty",
    "iter": "iter iterate iterator loop sequence",
    "next": "next iterate iterator advance",
    "call": "call invoke callable apply",
    "contains": "contains membership in has",
    "getitem": "getitem index lookup subscript key",
    "setitem": "setitem assign index subscript key",
    "enter": "enter context manager acquire setup",
    "exit": "exit context manager release teardown cleanup",
    "bool": "bool truthy falsy truth condition",
    "post_init": "post init initialize after construction defaults",
}
# `__x__` anywhere, or a QUOTED bare name ('repr'). The quote requirement on the
# bare form is deliberate: a ticket that merely uses the word "str" in prose must
# not drag six synonyms into the ranking query.
_DUNDER_RE = re.compile(r"__([a-z_]+)__")
_QUOTED_RE = re.compile(r"['\"`]([a-z_]+)['\"`]")


def expand_dunder_query(text):
    """`text` plus the lexical forms of any dunder it names. RETRIEVAL ONLY."""
    if not text:
        return text
    found = []
    for name in _DUNDER_RE.findall(text) + _QUOTED_RE.findall(text.lower()):
        words_for = DUNDER_LEXICAL.get(name)
        if words_for and words_for not in found:
            found.append(words_for)
    if not found:
        return text
    return text + "\n" + " ".join(found)


def file_rank_content_words(task_text, template_aware=True):
    """The ticket's OWN vocabulary, excluding fixed grounded-template prose.

    The template-aware subtraction is opt-out for the replay's old-ranker
    control: it is a ranking-only stoplist, so ordinary free-form tickets keep
    their existing vocabulary and the prompt text remains unchanged.
    """
    template_words = set()
    if template_aware and re.search(r"\baround\s+.+?\s*:", task_text or "", re.I):
        # SYMPTOM_TEMPLATE_WORDS is declared below with the ownership helpers;
        # lookup occurs at call time, after module initialization.
        template_words = SYMPTOM_TEMPLATE_WORDS
    return {w for w in re.findall(r"[a-z_][a-z0-9_]*", (task_text or "").lower())
            if w not in FILE_RANK_BOILERPLATE
            and w not in template_words and len(w) > 2}


# ---------------------------------------------------------------------------
# EVIDENCE-CARD HYGIENE (ladder autopsy 2026-08-29, ranked fix #5)
#
# The old card was `f"{p}  (best function: {fn_name} {str(why)[:60]})"`, and the
# traces show it doing three separate kinds of damage:
#
#   * `<module>` is a legal `best_func` value (func_index mints a module-level
#     pseudo-record). It was advertised as the file's "best function" in 27 of
#     84 runs. On a distractor it is harmless; on the GOLD file it tells the
#     model, correctly, that the file's best candidate cannot hold a bug.
#   * `str(why)` is the raw ranking dict, so `func_lex`/`parent_lex` FLOATS were
#     printed to the model — and in 47 of 84 traces the model read them as
#     program semantics: "the parameters are func_lex, parent_lex, construct_",
#     "'parent_lex' seems to be a parameter that might determine whether a case
#     should pass or fail". They are ranker internals and mean nothing to a
#     reader of the code.
#   * `[:60]` truncated mid-dict, so every card ended on a dangling
#     `'construct_count')` — exactly where the traces' mis-parses begin.
#
# So the card now shows only reader-meaningful evidence: the function name (never
# a pseudo-record) and the task words that actually matched, plus the construct
# count as a WORD. Nothing is truncated mid-token.
MODULE_PSEUDO_FUNCS = frozenset(("<module>", "module", ""))

# RANKED FIX #5 (rung-4000 autopsy, F4). The card said "best function: <name>"
# and the model read the SUPERLATIVE as a verdict on correctness, verbatim:
#   "This file has a function called `wrap_untrusted` which is described as the
#    best function for this task. It seems to be the correct function. So it's
#    not buggy."  -> {"pick": "B"}   (the gold file was A)
# "best" is a RETRIEVAL statement — this is the function that made the file rank
# — and the reader has no way to know that. "candidate" says the same thing
# without asserting anything about the code's quality. One word, one file-stage
# pick. This is the single label; nothing else may spell it.
CANDIDATE_FUNCTION_LABEL = "candidate function"


def _is_real_function(name):
    """False for func_index's module-level pseudo-record (and for nothing else)."""
    if not name:
        return False
    return str(name).split(".")[-1].strip().lower() not in MODULE_PSEUDO_FUNCS


# VOLUME IS NOT GUILT (armed-cycle correction, 2026-08-29). The function menu's
# evidence line ends "...; 25x bool_flip; 261 lines", and the model read that as
# an argument: chapter_report_card_service picked `_generate_and_upsert` over the
# gold function with the reasoning "25x bool_flip; 261 lines. This function is
# more complex and has many lines. It might be the one with the bug." Gold showed
# "2x bool_flip; 27 lines". A raw construct count and a line count are RETRIEVAL
# statistics — they say how much code there is, never that any of it is wrong,
# and a big function wins both mechanically. The counts are therefore normalized
# to a per-line DENSITY word and the raw line count is dropped: density is the
# quantity that actually discriminates, and it does not reward size.
_COUNT_RE = re.compile(r"(\d+)x\s+(\w+)")
_LINES_RE = re.compile(r";?\s*(\d+)\s+lines\b")


def _density_word(count, loc):
    """COARSE on purpose. The measured misread is a big function outscoring a
    small one on raw counts, and a fine-grained density would just reproduce the
    ranking in words: chapter_report_card's decoy is 25 sites / 261 lines = 9.6
    per 100 and the gold is 2 / 27 = 7.4 per 100, so a tight scale still hands
    the decoy the win. Three wide bands put both in the same bucket, which is the
    honest statement — neither density distinguishes them — and removes the
    spurious signal instead of dressing it up."""
    if not loc or count <= 0:
        return "none"
    per100 = 100.0 * count / float(loc)
    return "dense" if per100 >= 15 else "some" if per100 >= 5 else "sparse"


def sanitize_function_evidence(line):
    """Rewrite one func_index evidence line so it states DENSITY, not volume."""
    if not line:
        return line
    try:
        loc_m = _LINES_RE.search(line or "")
        loc = int(loc_m.group(1)) if loc_m else 0
        out = _LINES_RE.sub("", line or "")
        def _sub(m):
            return "%s %s" % (_density_word(int(m.group(1)), loc), m.group(2))
        out = _COUNT_RE.sub(_sub, out)
        return re.sub(r";\s*;", ";", out).strip().strip(";").strip()
    except Exception:
        return line


# ---------------------------------------------------------------------------
# CARD CONTENT (stacked-autopsy 'deeper cure', 2026-08-29)
#
# The cards were RENDERED honestly and SAID nothing. Measured on the armed
# cycle: a file card read `matched: and, for, the, that` — four stopwords, one
# per option, identical across the menu — and the model spent ~3k tokens looping
# on them because it had been handed evidence that could not discriminate and no
# statement of what any option WAS. Three defects, one cause (nothing on the card
# is about the code):
#
#   * the matched bag is raw lexical overlap, so English scaffolding dominates it
#     and every option shows the same words;
#   * nothing says what the file/function is FOR, so the only readable difference
#     between options is their position, which is E49 again;
#   * two options can render byte-identical text, and a menu with two identical
#     options has no correct answer.
#
# The cure is content, not formatting: stoplist-filtered matches ordered
# DISTINCTIVE-FIRST (a word held by one candidate outranks one held by all of
# them), a code-derived PURPOSE line, guaranteed-distinct visible text, and a
# length cap so none of it runs off. APPEND-ONLY: option ORDER never moves — only
# the descriptive text improves.
CARD_MAX_MATCHED = 4         # four content words say more than eight stopwords
CARD_MAX_CHARS = 160         # a card that wraps is a card that is skimmed
CARD_PURPOSE_CHARS = 80

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])(?:\s|$)")


def card_content_words(matched, df=None, limit=CARD_MAX_MATCHED):
    """The matched tokens worth showing: CONTENT words only, distinctive first.

    `df` is {lowercased word: how many candidates on this menu contain it}. A
    word every option shares discriminates nothing, so it sorts last; the sort is
    stable, so ties keep the ranker's own order. Filtering reuses
    FILE_RANK_BOILERPLATE — the same stoplist file_rank_content_words uses, so a
    word the ranker refuses to score is a word the card refuses to print.
    """
    out, seen = [], set()
    for w in matched or []:
        tok = str(w).strip()
        low = tok.lower()
        if not low or low in seen or len(low) <= 2 or low in FILE_RANK_BOILERPLATE:
            continue
        seen.add(low)
        out.append(tok)
    if df:
        out.sort(key=lambda t: int(df.get(t.lower(), 0)))
    return out[:max(0, limit)]


def menu_doc_frequency(matched_by_option):
    """{word: how many options list it} over one menu. Deterministic, no model."""
    df = {}
    for matched in (matched_by_option or []):
        for w in {str(x).strip().lower() for x in (matched or []) if str(x).strip()}:
            df[w] = df.get(w, 0) + 1
    return df


def _first_docstring_sentence(doc, limit=CARD_PURPOSE_CHARS):
    """The first sentence of a docstring, never cut mid-word."""
    text = re.sub(r"\s+", " ", (doc or "")).strip()
    if not text:
        return ""
    parts = _SENTENCE_END_RE.split(text, 1)
    sentence = (parts[0] if parts else text).strip()
    if len(sentence) > limit:
        cut = sentence[:limit].rsplit(" ", 1)[0].strip()
        sentence = (cut or sentence[:limit]).rstrip(",;:") + "..."
    return sentence


def _top_level_names(node, limit=3):
    """Up to `limit` defined names, in source order. Functions before classes:
    a bug lives in a function, so the functions are the informative half."""
    body = getattr(node, "body", None) or []
    fns = [x.name for x in body
           if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(fns) < limit:
        fns += [x.name for x in body if isinstance(x, ast.ClassDef)]
    return fns[:max(0, limit)]


def purpose_line(node, doc=None):
    """One PURPOSE clause for a card. CODE-DERIVED — never a model call.

    First docstring sentence when there is one, else what the thing DEFINES.
    Returns "" when neither is available; an empty clause is simply not printed.
    """
    try:
        if doc is None and node is not None:
            doc = ast.get_docstring(node)
    except Exception:
        doc = None
    sentence = _first_docstring_sentence(doc)
    if sentence:
        return "purpose: " + sentence
    names = _top_level_names(node) if node is not None else []
    return ("defines: " + ", ".join(names)) if names else ""


def module_purpose_line(src):
    """purpose_line for a whole file, from its source text. Fails open to ""."""
    if not src:
        return ""
    try:
        return purpose_line(ast.parse(src))
    except Exception:                        # unparseable file: say nothing
        return ""


def cap_card(text, limit=CARD_MAX_CHARS):
    """Hold a card to the line budget, dropping WHOLE clauses from the end.

    Never truncates mid-token: an option ending on a dangling fragment is the
    exact parse break the previous card generation was fixed for.
    """
    text = str(text or "")
    if len(text) <= limit:
        return text
    head, sep, body = text.partition("  (")
    if sep and body.endswith(")"):
        bits = body[:-1].split("; ")
        while len(bits) > 1 and len("%s  (%s)" % (head, "; ".join(bits))) > limit:
            bits.pop()
        out = "%s  (%s)" % (head, "; ".join(bits))
        if len(out) <= limit:
            return out
        text = out
    if len(text) <= limit:
        return text
    marker = " " + _TRUNCATION_MARKER
    room = max(1, int(limit) - len(marker))
    prefix = text[:room].rsplit(" ", 1)[0].strip() or text[:room]
    return prefix + marker


def distinguish_cards(keys, cards):
    """Guarantee no two options render identical visible text.

    Colliding cards are suffixed with the distinguishing segment of their own
    path — the deepest component that is not shared with the other colliders. A
    menu whose options read the same has no correct answer, and the model can
    only answer it by position. Order is NEVER touched.
    """
    out = list(cards)
    groups = {}
    for i, card in enumerate(out):
        groups.setdefault(card, []).append(i)
    for card, idxs in groups.items():
        if len(idxs) < 2:
            continue
        parts = [[p for p in str(keys[i] or "").split("/") if p] for i in idxs]
        for pos, i in enumerate(idxs):
            mine, others = parts[pos], [p for j, p in enumerate(parts) if j != pos]
            seg = next((s for s in reversed(mine)
                        if all(s not in o for o in others)), None)
            if seg is None:                  # genuinely the same candidate
                seg = "/".join(mine[-2:]) if mine else ""
            if seg:
                out[i] = "%s [%s]" % (card, seg)
    return out


_SYMPTOM_WORDS_RE = re.compile(r"(\d+)\s+symptom words(?::\s*([^;]*))?")


def symptom_words_of(line):
    """The words a func_index evidence line lists, raw. [] when it lists none."""
    m = _SYMPTOM_WORDS_RE.search(line or "")
    if not m or not m.group(2):
        return []
    return [w.strip() for w in m.group(2).split(",") if w.strip()]


def filter_symptom_words(line, df=None):
    """Rewrite a func_index evidence line so its word list is CONTENT words only.

    The count is recomputed from what survives: a card that says "6 symptom
    words: and, for, the" is claiming evidence it does not have, and the count is
    read as a score (it is what lost `create_session`).
    """
    def _sub(m):
        kept = card_content_words(
            [w.strip() for w in (m.group(2) or "").split(",") if w.strip()], df)
        if not kept:
            return "no distinctive symptom words"
        return "%d symptom word%s: %s" % (len(kept), "" if len(kept) == 1 else "s",
                                          ", ".join(kept))
    try:
        return _SYMPTOM_WORDS_RE.sub(_sub, line or "")
    except Exception:
        return line


_TRUNCATION_MARKER = "...[truncated]"


def _truncate_for_prompt(value, limit):
    """Middle-elide prompt data and make every cut machine-visible."""
    text = "" if value is None else str(value)
    cap = max(1, int(limit))
    if len(text) <= cap:
        return text
    marker = _TRUNCATION_MARKER
    if cap <= len(marker):
        return marker
    room = cap - len(marker)
    left = (room + 1) // 2
    right = room // 2
    return text[:left] + marker + (text[-right:] if right else "")


def _truncate_code_change(value, limit):
    """Middle-elide a code change while retaining its operator when possible."""
    text = "" if value is None else str(value)
    cap = max(1, int(limit))
    if len(text) <= cap:
        return text
    marker = _TRUNCATION_MARKER
    match = re.search(r"(==|!=|<=|>=|<|>|\bis\b|\bin\b)", text)
    if match and cap > len(marker) * 2 + len(match.group(0)) + 2:
        room = cap - len(marker) * 2 - len(match.group(0))
        left = max(1, (room + 1) // 2)
        right = max(1, room // 2)
        return (text[:left] + marker + match.group(0) + marker
                + (text[-right:] if right else ""))
    return _truncate_for_prompt(text, cap)


def _prompt_json(value, limit=None):
    """Render untrusted prompt data as one JSON string, optionally marked/capped."""
    if limit is not None:
        value = _truncate_for_prompt(value, limit)
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def probe_snippet(src, query_words, max_lines=6, max_chars=480, focus=None):
    """The lines of `src` that put a FILE on the menu, rendered for a binary probe.

    PURE and deliberately small. A binary file probe must see the evidence, not
    the file: N probes x a whole module is both unaffordable and a different
    question (recall, not relevance). So it gets the same thing the evidence card
    was built from — the lines carrying the ticket's own terms — with the line
    numbers kept so a reader of the certificate can go and look.

    Selection is by MATCH COUNT, ties broken by file order, and the OUTPUT is
    always re-sorted into file order: a probe reading top-scoring lines out of
    sequence is being shown a ranking, and it would score the ranking.

    `focus` (the function whose evidence made the file rank) seeds the window so
    a file that ranked on a function name rather than on body words still shows
    something. An empty result is rendered explicitly, never as blank space — a
    probe answering about nothing must be able to say so.
    """
    lines = (src or "").splitlines()
    qs = {str(w).lower() for w in (query_words or []) if w}
    if focus:
        qs.add(str(focus).split(".")[-1].lower())
    scored = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        low = ln.lower()
        hits = sum(1 for w in qs if w and w in low)
        if hits:
            scored.append((-hits, i))
    scored.sort()
    max_lines = max(0, int(max_lines))
    keep = sorted(i for _h, i in scored[:max_lines])
    line_limited = len(scored) > max_lines
    if not keep:
        out = ["    (no line in this file mentions the problem's terms)"]
        if line_limited:
            out.append("  ...[truncated]")
        return "\n".join(out)
    out, used = [], 0
    for i in keep:
        row = "  %5d: %s" % (i + 1,
                            _truncate_for_prompt(lines[i].strip(), 120))
        if used + len(row) > max_chars:
            out.append("  ...[truncated]")
            break
        out.append(row)
        used += len(row)
    if line_limited and (not out or out[-1] != "  ...[truncated]"):
        out.append("  ...[truncated]")
    return "\n".join(out)


def function_probe_body(src, node, max_lines=14, max_chars=700):
    """A FUNCTION candidate rendered for a binary probe: signature + head of body.

    Same contract as `probe_snippet` — bounded, ordered, and honest about what it
    left out. Truncation is MARKED, so a probe answered on half a function can
    never be mistaken in the ledger for one answered on the whole thing.
    """
    if node is None:
        return "    (source unavailable)"
    lines = (src or "").splitlines()
    lo = max(0, getattr(node, "lineno", 1) - 1)
    hi = min(len(lines), getattr(node, "end_lineno", None) or (lo + 1))
    span = lines[lo:hi]
    clipped = len(span) > max_lines
    span = span[:max(1, int(max_lines))]
    out, used = [], 0
    for j, ln in enumerate(span):
        row = "  %5d: %s" % (lo + j + 1,
                            _truncate_for_prompt(ln.rstrip(), 120))
        if used + len(row) > max_chars:
            clipped = True
            break
        out.append(row)
        used += len(row)
    if clipped:
        out.append("  ...[truncated]")
    return "\n".join(out) if out else "    (source unavailable)"


def function_card(name, ev_line, node=None, df=None):
    """One menu display line for a FUNCTION candidate: density not volume,
    content words not stopwords, and a code-derived statement of purpose."""
    bits = []
    purpose = purpose_line(node) if node is not None else ""
    if purpose:
        bits.append(purpose)
    body = filter_symptom_words(sanitize_function_evidence(ev_line), df)
    if body:
        bits.append(body)
    return cap_card("%s  (%s)" % (name, "; ".join(bits)) if bits else str(name))


def evidence_card(path, ev, matched_fallback, cls=None, src=None, df=None):
    """One menu display line for a file candidate. Never prints a ranking float,
    never advertises `<module>` as a function, never truncates mid-dict, and
    never shows a bag of stopwords in place of evidence."""
    why = ev.get("why") if isinstance(ev, dict) else None
    matched = []
    constructs = 0
    if isinstance(why, dict):
        try:
            matched = [str(w) for w in (why.get("matched_idents") or [])]
            constructs = int(why.get("construct_count") or 0)
        except (TypeError, ValueError):
            matched, constructs = [], 0
    if not matched:
        matched = list(matched_fallback or [])
    matched = card_content_words(matched, df)
    bits = []
    purpose = module_purpose_line(src)
    if purpose:
        bits.append(purpose)
    bits.append("matched: " + (", ".join(matched) if matched else "-"))
    if constructs and cls:
        # a WORD, not a count: "12x cmp_flip" is volume, and volume-as-guilt is
        # the measured misread this card exists to remove.
        bits.append(f"has {cls} sites")
    fn_name = ev.get("best_func") if isinstance(ev, dict) else None
    if _is_real_function(fn_name):
        # keep the whole qualname; a name cut mid-token is the parse break above
        bits.insert(0, f"{CANDIDATE_FUNCTION_LABEL}: "
                       f"{str(fn_name).split('.')[-1]}")
    return cap_card(f"{path}  ({'; '.join(bits)})")


def card_matched_words(ev, matched_fallback):
    """The RAW matched tokens a card would draw on — the input to the menu-wide
    document-frequency count, computed before any card is rendered."""
    why = ev.get("why") if isinstance(ev, dict) else None
    if isinstance(why, dict):
        try:
            got = [str(w) for w in (why.get("matched_idents") or [])]
            if got:
                return got
        except (TypeError, ValueError):
            pass
    return [str(w) for w in (matched_fallback or [])]


def _module_tail(rel_path):
    """The dotted tail a corpus path is importable as. ``None`` if it has none.

    ``utils/dispatch.py`` -> ``utils.dispatch``; ``utils/__init__.py`` ->
    ``utils``. Deliberately a TAIL and not a full module name: the corpus root
    is `backend/app`, so the real import reads `app.utils.dispatch` and only the
    tail is knowable from the path alone.
    """
    if not rel_path.endswith(".py"):
        return None
    stem = rel_path[:-3].replace("\\", "/")
    parts = [x for x in stem.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def import_fan_in(files):
    """``{rel_path: distinct in-corpus importers}``. Deterministic, no I/O.

    THE ANTI-ATTRACTOR SIGNAL. A file that half the
    application imports is a HUB: on a scrubbed symptom BM25 ranks on the
    corpus's own term mass and the file menu collapses onto exactly those hubs
    regardless of the ticket. Measured shape of that failure on the fresh-21
    board: `utils/attempt_context.py` was the file-support gate's subject on
    all six abstains, every time, for six unrelated tickets.

    COUNTING RULES, all from the design and all load-bearing:
      * ONE edge per (importer, target) pair — not per import statement, not
        per imported name, not per call site. Two `from x import a` /
        `from x import b` lines are one importer.
      * a file never imports itself;
      * an import that resolves to nothing in this corpus is EXCLUDED, not
        counted as an edge to nowhere;
      * TEST FILES ARE ALREADY EXCLUDED upstream — `load_repo` drops any path
        whose name contains `test` — so this needs no test filter of its own,
        and adding one would be a second, drifting authority.

    Ambiguity is resolved by LONGEST TAIL and never by guessing: if two corpus
    files' dotted tails both suffix-match one import, the longer (more specific)
    one wins, and an exact tie credits neither. A fan-in count that guessed
    would penalise an innocent file, which is the exact failure mode this
    signal exists to avoid.
    """
    tails = {}
    for rel in files:
        t = _module_tail(rel)
        if t:
            tails.setdefault(t, []).append(rel)
    edges = set()
    for importer, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                       # unparseable file casts no votes
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.level:
                    mods.add(node.module)
                    # `from pkg.mod import name` may also name a MODULE, which
                    # is an edge to `pkg/mod/name.py` and not to `pkg/mod.py`.
                    for a in node.names:
                        mods.add(node.module + "." + a.name)
        for m in mods:
            best, best_len = None, -1
            for t, rels in tails.items():
                if (m == t or m.endswith("." + t)) and len(t) > best_len:
                    best, best_len = rels, len(t)
            if best is None or len(best) != 1:
                continue                   # unresolved, or ambiguous -> no edge
            target = best[0]
            if target != importer:
                edges.add((importer, target))
    out = {}
    for _imp, tgt in edges:
        out[tgt] = out.get(tgt, 0) + 1
    return out


# =================================================== OWNERSHIP TIE-BREAKERS
# Ownership tie-breakers were wired 2026-08-29.
#
# THE PROBLEM THE FAN-IN PRIOR ALONE DOES NOT SOLVE. Measured by
# `probes/locate_fan_in_replay.py` on a faithful per-repo corpus: baseline
# gold-top1 is 9/21 and the best feasible (K, m) buys exactly ONE row. Fan-in
# says a file is a hub; it does not say the OTHER file owns the symptom. On a
# scrubbed symptom the attractor and the gold often share the same use-site
# vocabulary, and the thing that separates them is WHERE those tokens live:
#
#     the gold DEFINES the symptom's nouns;
#     the attractor merely USES them.
#
# So the signal is ownership, and fan-in is one damped term inside it:
#
#     H_eff = H * (1 - D)^2
#     S     = 60B + 15D + 5U + 12V + 3C - 10 H_eff
#
# The SQUARED ownership guard is the whole trick. Hubness is nearly free for a
# file that genuinely defines the symptom's vocabulary (D high => (1-D)^2 near
# zero) and expensive for a shared utility whose overlap is inherited use-site
# vocabulary. That is what keeps this a tie-breaker rather than a blanket
# "utilities are never buggy" rule — a real defect in a real hub keeps its D.
#
# NOT AN EXCLUSION, EVER: this reorders `cand` and never changes membership, so
# pool recall is untouched by construction. Hard anchors are exempt (a named
# symbol is proof; a prior may not outrank proof — #197).

#: Tokens that are everywhere in this corpus and own nothing. Capped at 0.25 of
#: normal weight per the design, NOT dropped: `status` in a symptom is weak
#: evidence, not zero evidence, and dropping it would let a symptom made only of
#: generic words produce an empty vocabulary and a vacuous 100% coverage.
GENERIC_TOKENS = frozenset((
    "context", "data", "attempt", "run", "status", "service", "utils", "util",
    "get", "set", "value", "result", "config", "app", "model", "type", "list",
    "item", "key", "name", "id", "obj", "self", "args", "kwargs", "res",
    "response", "request", "session", "db", "log", "logger", "error", "async",
    "def", "class", "return", "task", "job", "info", "count", "index", "text",
))
GENERIC_WEIGHT_FRAC = 0.25

#: Definition : use : import. From the design, verbatim. An import alone must
#: never look like ownership — that is exactly how a barrel file steals a gold.
W_DEF, W_USE, W_IMPORT = 1.00, 0.25, 0.10


#: The GROUNDED-SYMPTOM TEMPLATE. `probes/mr_harvest2.py` emits every ticket on
#: this board as
#:
#:     "Something is going wrong around <n1>, <n2>, <n3> and <n4>: <one-line>"
#:
#: and the noun list is the ONLY grounded content in it. Everything else is
#: fixed scaffolding that appears in all 21 tickets identically.
_SYMPTOM_TEMPLATE_RE = re.compile(r"\baround\s+(.+?)\s*:", re.I)

#: Words the template itself contributes. MEASURED, not guessed: on t2 the
#: symptom yields 13 content words of which 9 are these, so BOILERPLATE IS 69%
#: OF THE TOKEN MASS. Any coverage-normalised feature (D, U, and the IDF
#: evidence channel) divides by a total that the gold file can never match,
#: which is why measured D for the gold was 0.01 while the ranking gap it had to
#: close was six points of fused score. `FILE_RANK_BOILERPLATE` predates this
#: template and contains none of these.
SYMPTOM_TEMPLATE_WORDS = frozenset((
    "something", "going", "around", "outcome", "way", "spans", "several",
    "steps", "routine", "reaches", "reached", "happening", "happen",
    "processed", "anyway", "rejected", "caught", "raised", "property",
    "object", "right", "dependency", "fixed", "used", "one", "step",
    "input",
))


def symptom_subject_nouns(task_text):
    """The ticket's GROUNDED nouns — its only real content — or ``None``.

    Returns ``None`` (never an empty set, and never a silent fallback) when the
    text is not in the grounded template, so a caller can tell "this ticket has
    no extractable subject list" from "its subject list is empty". A silent
    fallback here would make an unparsed ticket look like a parsed one whose
    nouns happened to match nothing, which is the fail-plausible pattern.
    """
    m = _SYMPTOM_TEMPLATE_RE.search(task_text or "")
    if not m:
        return None
    body = m.group(1).replace(" and ", ", ")
    out = set()
    for chunk in body.split(","):
        for w in split_identifier(chunk.strip()):
            if len(w) > 2 and w not in SYMPTOM_TEMPLATE_WORDS:
                out.add(w)
    return out or None


def symptom_vocabulary(task_text):
    """The token set the OWNERSHIP features score against.

    Grounded nouns when the ticket is in the template, otherwise the general
    content words minus the template scaffolding. One authority, so D, U and the
    hubness damping all divide by the same denominator.
    """
    nouns = symptom_subject_nouns(task_text)
    if nouns:
        return nouns
    return {w for w in file_rank_content_words(task_text)
            if w not in _LINKAGE_STOP and w not in SYMPTOM_TEMPLATE_WORDS}


def _tok_weight(tok, idf):
    w = float((idf or {}).get(tok, 1000))
    return w * (GENERIC_WEIGHT_FRAC if tok in GENERIC_TOKENS else 1.0)


def file_symbol_sites(src):
    """``(defined, used, imported)`` token sets for one file. Never raises.

    DEFINED  names this file BINDS at module or class level: function names,
             class names, module-level assignment targets, and the tokens of
             route/decorator string literals (a `@router.get("/lessons")` is a
             definition of `lessons` in every sense that matters here).
    USED     executable references the body READS — `Name` loads, attribute
             names, call targets — minus anything this file defines.
    IMPORTED module tails and imported names.

    The split IS the signal, so an unparseable file yields three empty sets
    rather than a guess: an ownership score computed from a failed parse would
    be indistinguishable from an honest zero.
    """
    defined, used, imported = set(), set(), set()
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return frozenset(), frozenset(), frozenset()

    def add(dst, raw):
        for w in split_identifier(raw):
            if len(w) > 2:
                dst.add(w)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(defined, node.name)
            for d in node.decorator_list:
                for sub in ast.walk(d):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        add(defined, sub.value.replace("/", "_").replace("-", "_"))
        elif isinstance(node, ast.ClassDef):
            add(defined, node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                add(imported, a.name.replace(".", "_"))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                add(imported, node.module.replace(".", "_"))
            for a in node.names:
                add(imported, a.name)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in ([node.target] if isinstance(node, ast.AnnAssign)
                      else node.targets):
                if isinstance(t, ast.Name):
                    add(defined, t.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            add(used, node.id)
        elif isinstance(node, ast.Attribute):
            add(used, node.attr)
    used -= defined
    imported -= defined
    return frozenset(defined), frozenset(used), frozenset(imported)


def ownership_features(sym_tokens, sites, idf):
    """``(D, U)`` in [0, 1]: weighted symptom coverage by DEFINED / USED tokens.

    Coverage, not raw overlap. A big file contains more of everything, so an
    unnormalised count would rank on file size — which is the failure the
    length-norm term already exists to fight one level up.
    """
    if not sym_tokens:
        return 0.0, 0.0
    defined, used, imported = sites
    total = sum(_tok_weight(t, idf) for t in sym_tokens) or 1.0
    d = sum(_tok_weight(t, idf) for t in sym_tokens & defined)
    u = sum(_tok_weight(t, idf) for t in sym_tokens & used)
    i = sum(_tok_weight(t, idf) for t in sym_tokens & imported)
    D = (W_DEF * d) / total
    U = (W_USE * u + W_IMPORT * i) / total
    return min(1.0, D), min(1.0, U)


#: SYMPTOM VERB -> compatible FILE ROLE. Frozen, code-derived on the file side
#: (never a filename guess): a role is claimed by what a file DOES.
ROLE_EVIDENCE = {
    "scheduler": ("schedule", "cron", "interval", "next_run", "due", "tick",
                  "pulse", "heartbeat", "enqueue", "dispatch"),
    "worker": ("consume", "poll", "worker", "queue", "sqs", "background"),
    "validator": ("validate", "valid", "reject", "raise", "assert", "check"),
    "serializer": ("serialize", "to_dict", "json", "dump", "render", "format"),
    "mapper": ("map", "convert", "transform", "normalize", "projection"),
    "query": ("query", "select", "filter", "session", "fetch", "load"),
    "guard": ("permission", "authorize", "scope", "tenant", "workspace",
              "forbidden"),
}
SYMPTOM_ROLES = {
    # symptom verb -> (preferred roles, adjacent roles)
    "stale":    (("scheduler", "worker"), ("query",)),
    "outdated": (("scheduler", "worker"), ("query",)),
    "refresh":  (("scheduler", "worker"), ("query",)),
    "duplicate": (("worker", "guard"), ("query",)),
    "repeated": (("worker", "guard"), ("query",)),
    "delayed":  (("scheduler", "worker"), ()),
    "missed":   (("scheduler", "worker"), ()),
    "missing":  (("query", "mapper"), ("serializer",)),
    "omitted":  (("query", "mapper"), ("serializer",)),
    "invalid":  (("validator",), ("mapper",)),
    "rejected": (("validator", "guard"), ()),
    "malformed": (("validator", "mapper"), ()),
    "format":   (("serializer", "mapper"), ()),
    "shape":    (("serializer", "mapper"), ()),
    "wrong":    ((), ()),
}


def file_roles(src):
    """Roles a file CLAIMS BY ITS CODE. Evidence, not filename inference."""
    low = (src or "").lower()
    return frozenset(r for r, marks in ROLE_EVIDENCE.items()
                     if sum(1 for m in marks if m in low) >= 2)


def role_compat(task_text, roles):
    """`V` in [0, 1]. NEUTRAL 0.5 when no verb is recognised.

    Neutral, not zero. An unrecognised verb is an absence of evidence, and
    scoring it 0 would silently punish every file for the classifier's gap —
    an unmeasured channel must never become a finding (the same rule the
    file-support gate obeys when its structural check cannot run).
    """
    low = (task_text or "").lower()
    pref, adj = set(), set()
    seen = False
    for verb, (p, a) in SYMPTOM_ROLES.items():
        if verb in low:
            seen = True
            pref |= set(p)
            adj |= set(a)
    if not seen or not (pref or adj):
        return 0.5
    if roles & pref:
        return 1.0
    if roles & adj:
        return 0.5
    return 0.0


def hubness(fan_in, paths):
    """`H` in [0, 1]: log fan-in normalised between the menu's P50 and P95.

    Percentile-normalised rather than thresholded, because "how much of a hub"
    is corpus-relative: a fixed K that is a hub in one repo is unremarkable in
    another. ``P95 <= P50`` (a flat menu, where hubness separates nothing)
    yields H=0 for everyone — the design's own rule, and the right one: a
    feature that cannot discriminate must be inert, not arbitrary.
    """
    import math
    vals = sorted(int((fan_in or {}).get(p, 0)) for p in paths)
    if not vals:
        return {}
    def pct(q):
        return vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))]
    p50, p95 = pct(0.50), pct(0.95)
    lo, hi = math.log(1 + p50), math.log(1 + p95)
    if hi <= lo:
        return {p: 0.0 for p in paths}
    return {p: max(0.0, min(1.0, (math.log(1 + int((fan_in or {}).get(p, 0)))
                                  - lo) / (hi - lo)))
            for p in paths}


#: The design's coefficients, verbatim. `C` (git churn) is DELIBERATELY ABSENT:
#: it requires a report-time history cutoff this harness cannot honour offline,
#: and the design's own rule for a missing index is "neutral, not guessed". A
#: neutral C is a constant added to every candidate, which cannot change an
#: ordering — so omitting it and adding a fabricated one are the same ranking,
#: and only one of them is honest about what was measured.
OWN_W_B, OWN_W_D, OWN_W_U, OWN_W_V, OWN_W_H = 60.0, 15.0, 5.0, 12.0, 10.0


def rank_file_menu(cand, func_evidence, idx, task_text, fn_anchor_files, cfg,
                   fan_in=None, files=None, idf=None, baseline=None,
                   compose=True):
    """Reorder `cand` by fused evidence. Returns a PERMUTATION of `cand`.

    cand            the candidate paths, already ordered by the shipping code
    func_evidence   {path: evidence dict} from file_menu_from_functions; carries
                    adj_score and the matched identifiers
    idx             {path: {word: count}} file index (for the length term)
    fn_anchor_files paths holding an exact-symbol FUNCTION anchor
    """
    if not cand or not getattr(cfg.locate, "file_rank_fusion", False):
        return list(cand)
    b = float(getattr(cfg.locate, "file_rank_lennorm_b", 0.0) or 0.0)
    cw_weight = float(getattr(cfg.locate, "file_rank_content_w", 0.0) or 0.0)
    definition_weight = float(
        getattr(cfg.locate, "file_rank_definition_w", 0.0) or 0.0)
    try:
        definition_min = max(0.0, min(1.0, float(
            getattr(cfg.locate, "file_rank_definition_min", 0.0) or 0.0)))
    except (TypeError, ValueError):
        definition_min = 0.0
    definition_guard = bool(
        getattr(cfg.locate, "file_rank_definition_guard", False))
    boost = float(getattr(cfg.locate, "file_rank_fn_anchor_boost", 0.0) or 0.0)
    sizes = {p: sum((idx.get(p) or {}).values()) for p in cand}
    avg = max(1.0, sum(sizes.values()) / max(1, len(sizes)))
    content = (file_rank_content_words(
        task_text,
        template_aware=bool(
            getattr(cfg.locate, "file_rank_template_stoplist", True)
            and len(cand) > MAX_MODEL_FILE_MENU_OPTIONS))
               if cw_weight else set())
    anchors = set(fn_anchor_files or ())
    _definition = None
    if definition_weight and files and len(cand) > MAX_MODEL_FILE_MENU_OPTIONS:
        try:
            _sym = symptom_vocabulary(task_text)
            _idf = idf if idf is not None else build_idf(files)
            _definition = {}
            for p in cand:
                _definition[p] = ownership_features(
                    _sym, file_symbol_sites(files.get(p, "")), _idf)[0]
        except Exception:                    # law 1: fail OPEN to old rank
            _definition = None
    # `fan_in_penalty` OFF (the shipped default) leaves `_fan_m` None, and the
    # scoring loop below is then byte-identical to the pre-fan-in ranker. That
    # is the whole reason the knob reads as a tri-state here rather than as a
    # multiplier defaulting to 1.0: `s *= 1.0` is not free in float terms and a
    # dormant experiment must be provably inert, not approximately inert.
    _fan_m = None
    if getattr(cfg.locate, "fan_in_penalty", False):
        _fan_m = float(getattr(cfg.locate, "fan_in_m", 1.0) or 1.0)
        _fan_k = int(getattr(cfg.locate, "fan_in_k", 0) or 0)
    # ---- OWNERSHIP ARM ---------------------------------------------------
    # Replaces the multiplicative fused score with the additive S of
    # `locate-attractor-tie-breakers.md` for NON-ANCHORED candidates. It needs
    # the corpus (`files`) to see WHERE a token lives; without it the arm is
    # inert rather than approximated, because a D computed from the word-bag
    # index would count a use-site as a definition and invert the whole signal.
    _own = None
    if getattr(cfg.locate, "ownership_rank", False) and files:
        try:
            _sym = symptom_vocabulary(task_text)
            _idf = idf if idf is not None else build_idf(files)
            _H = hubness(fan_in or {}, cand)
            _own = {}
            for p in cand:
                _sites = file_symbol_sites(files.get(p, ""))
                _D, _U = ownership_features(_sym, _sites, _idf)
                _V = role_compat(task_text, file_roles(files.get(p, "")))
                _own[p] = (_D, _U, _V, _H.get(p, 0.0))
        except Exception:                    # law 1: fail OPEN to the old rank
            _own = None
    # `B` must be NORMALISED before it can be added to features in [0, 1], or
    # the 60B term is on an arbitrary scale and the other coefficients are
    # decoration. Max-normalised over this menu, so B is comparable within the
    # decision it is actually used for.
    _bmax = 0.0
    if _own is not None:
        for p in cand:
            try:
                _bmax = max(_bmax, float(
                    (func_evidence.get(p) or {}).get("adj_score") or 0.0))
            except (TypeError, ValueError):
                pass
    scored = []
    for i, p in enumerate(cand):
        ev = func_evidence.get(p) or {}
        try:
            s = float(ev.get("adj_score") or 0.0)
        except (TypeError, ValueError):
            s = 0.0
        if b:
            # BM25's missing document-length term, at file granularity. Floored
            # so a pathologically large file is damped, never zeroed.
            s /= max(0.2, (1.0 - b) + b * (sizes.get(p, avg) / avg))
        if cw_weight:
            matched = (ev.get("why") or {}).get("matched_idents") or []
            s *= 1.0 + cw_weight * len(set(matched) & content)
        if (_definition is not None and p not in anchors
                and _definition.get(p, 0.0) >= definition_min):
            # Definition-site coverage is evidence, not a replacement score:
            # cap it through the configured multiplier so a file with a generic
            # use-site cannot outrank a materially stronger lexical match.
            s *= 1.0 + definition_weight * _definition.get(p, 0.0)
        if p in anchors:
            s *= 1.0 + boost
        # ---- FAN-IN ANTI-ATTRACTOR PENALTY ---------------------
        # A RANKING PRIOR AND NOTHING ELSE. It never removes a candidate from
        # the pool, never changes menu MEMBERSHIP, and never fires on a HARD
        # ANCHOR: the ticket naming a function this file defines is proof, and
        # demoting a proof with a prior would trade evidence for a heuristic —
        # the wrong direction (#197).
        #
        # KNOWN FAILURE MODE, stated because it is real and not hypothetical:
        # when the bug genuinely IS in the shared utility, this penalises the
        # gold by design. That is why it is bounded (one multiplier, applied
        # once, above a threshold), why anchors are exempt, and above all why
        # the calibration below is gated on ZERO regressions among the tasks
        # that are currently correct — not on maximising conversions.
        if _fan_m is not None and p not in anchors:
            if int((fan_in or {}).get(p, 0)) > _fan_k:
                s *= _fan_m
        if _own is not None and p not in anchors:
            # H_eff = H(1-D)^2 — hubness is cheap for a real definition site.
            _D, _U, _V, _H = _own[p]
            _B = (s / _bmax) if _bmax > 0 else 0.0
            _heff = _H * (1.0 - _D) ** 2
            s = (OWN_W_B * _B + OWN_W_D * _D + OWN_W_U * _U + OWN_W_V * _V
                 - OWN_W_H * _heff)
        elif _own is not None:
            # An anchored file is EXEMPT and must still outrank every scored
            # candidate: mixing a raw fused score into an S-scaled menu would
            # sort a proof below a prior purely on units. Its S is the ceiling.
            s = OWN_W_B + OWN_W_D + OWN_W_U + OWN_W_V + 1.0
        # index i is the tie-break, so an all-zero-evidence menu keeps exactly
        # the order it arrived in (fail-safe: no evidence => no reordering).
        scored.append((-s, i, p))
    scored.sort()
    out = [p for _, _, p in scored]
    # NON-NEGOTIABLE: permutation only. If this ever fails, keep the incumbent
    # order rather than shipping a menu that lost a candidate.
    if sorted(out) != sorted(cand):
        return list(cand)
    if not compose:
        incumbent = baseline[0] if baseline else (cand[0] if cand else None)
        if (definition_guard and len(cand) > MAX_MODEL_FILE_MENU_OPTIONS
                and incumbent in out
                and incumbent not in out[:MAX_MODEL_FILE_MENU_OPTIONS]):
            # The wide raw arm may promote a strong definition site, but it must
            # not make the incumbent's best evidence unreachable in the primary
            # menu. This is an evidence-retention guard, not a gold-path prior.
            out.remove(incumbent)
            out.insert(MAX_MODEL_FILE_MENU_OPTIONS - 1, incumbent)
        return out
    return _joint_rerank(
        list(baseline) if baseline is not None else list(cand),
        proposed_orders=(out,), pinned=fn_anchor_files,
        mode=_rerank_compose_mode(cfg),
        max_shift=_rerank_compose_budget(cfg))

FIX_LINE_Q = """Function `{fn}` source context:
{region}

Line {line_no} (the line to fix):
{target}

Rewrite line {line_no} so that {task}
Output only the replacement source code for line {line_no}; do not output a line number, explanation, diagnostic, or code fence."""

LINE_MENU_Q = """TASK: {task}
Function `{fn}`:
{menu}
Which line is wrong? Reply with only one displayed menu letter."""

# POSITION-FREE LINE BALLOT (measured 2026-08-28, cfg 543b771d3cd4 autopsy).
# The mutation MENU is not a decision: across the last honest cycle the model
# resolved to option A in 11 of the 12 menus it answered, and `fixed` correlated
# perfectly with "gold happened to be at index 0" — it even picked a DOCSTRING
# prose line with margin 2.19 over the obvious boolean bug. Menu position, not
# content, was deciding every repair. Reversing the menu (menu_pick's existing
# thin-margin retry) only detects that bias, it cannot convert it: a purely
# position-locked answerer disagrees with itself and honestly abstains.
# So the ballot asks a question that HAS no position dimension: one independent
# yes/no call per distinct candidate line. Scores are comparable across calls,
# so the winner is chosen by content. Measured on the 18 tasks whose gold
# candidate was enumerable: the ballot names the correct line 13/18 where the
# menu's index-0 lands on it 8/18.
LINE_BALLOT_Q = """TASK: {task}
The bug is a single wrong line in function `{fn}`.
Numbered source context:
{context}

Candidate line id `{candidate_id}`:
Is the bug on THIS line?

    {line}

Judge candidate line `{candidate_id}` against the function context and the
described behaviour, not its position in the context.

Reply with only: {{"bug_here": "<yes|no>"}}  (replace the placeholder)"""


def _line_ballot_context(lines, lo, hi, candidate_line, max_lines=11,
                         max_chars=900):
    """Render candidate context with gutters and a visible cap."""
    start = max(int(lo), int(candidate_line) - 3)
    end = min(int(hi), int(candidate_line) + 4)
    all_indices = list(range(start, end))
    indices = list(all_indices)
    if int(lo) not in indices:
        indices.insert(0, int(lo))
    indices = list(dict.fromkeys(indices))
    max_lines = max(1, int(max_lines))
    line_limited = len(indices) > max_lines
    indices = indices[:max_lines]
    rows = []
    for index in indices:
        marker = "  <candidate>" if index == int(candidate_line) else ""
        source = _truncate_for_prompt(lines[index].rstrip(), 120)
        rows.append("  %4d%s | %s" % (index + 1, marker, source))
    if line_limited:
        rows.append("  ...[truncated]")
    return _truncate_for_prompt("\n".join(rows), max_chars)

# CANDIDATE-GRANULARITY BINARY PROBE (conversion autopsy 2026-08-29, §3).
# The line ballot above asks about a bare LINE. This asks about the actual EDIT.
# That distinction is the whole build: on `ea639fc1` the line-granularity ballot
# scored the gold LINE at -0.9573 while the decoy scored +0.6754, but the gold
# line reads `content = msg.message[:120]...` — nothing about the bare line says
# it is wrong. Rendering `msg.message -> msg.content` is a materially different
# question, and it is the one the selector actually needs answered.
#
# Deliberately NOT a menu: no letters, no options list, no position dimension.
# One candidate, one call, a score comparable across calls via `_yes_no_score`.
#
# WORDING IS CALIBRATION. "Does this
# fix the bug?" / "is this correct?" are approval questions: they invite the
# agreement behaviour that produced 34 "yes" out of 40 on the line ballot. The
# question below is instead narrow, causal and falsifiable — it asks whether a
# MECHANISM is visible in the edit — and it states the rule that disqualifies an
# edit which merely moves code around. No word in it ("correct", "right", "fix
# this", "gold") leaks the label the ranker is looking for.
CAND_PROBE_Q = """OBSERVED PROBLEM: {task}

Proposed change to `{fn}`:

    before:  {old}
    after:   {new}

Answer yes only if this change alters the behaviour described above. A change
that only renames, reformats, or moves code without changing what the function
does is a no.

Is there a visible mechanism in this change that prevents the observed problem?

Return exactly yes or no. Do not explain.
Reply with only: {{"fixes_bug": "<yes|no>"}}  (replace the placeholder)"""

# CONFIRM/VETO (E49: code ranks, the model confirms). Asked ONCE, about the
# candidate the probe scores already ranked first. Its only power is to say NO —
# a negative answer turns a ranked winner into an honest ABSTAIN. It cannot
# promote anything, cannot reorder, and an unreadable answer changes nothing.
#
# The confirm deliberately does NOT say "highest-ranked", "best" or "chosen"
# Telling the model something already won is exactly the approval
# frame that makes a veto unreachable. It is asked the INVERSE question —
# whether the problem SURVIVES the change — so a sycophantic "yes" argues
# against applying rather than for it, and the polarity is flipped in code.
CAND_CONFIRM_Q = """OBSERVED PROBLEM: {task}

Proposed change to `{fn}`:

    before:  {old}
    after:   {new}

After this change is applied, can the observed problem still happen?

Return exactly yes or no. Do not explain.
Reply with only: {{"still_broken": "<yes|no>"}}  (replace the placeholder)"""

# ===========================================================================
# BINARY PROBES FOR THE **FILE** AND **FUNCTION** STAGES (E100 port).
#
# Same mechanism as CAND_PROBE_Q above, ported to the two stages that still ran
# the slot-letter menu and still A-locked: E100 measured the file stage picking
# index 0 twice with gold at index 3, and the results ledger carries the same
# shape in every row since 1788010000. A probe sees ONE candidate, so there is
# no menu order for the answer to track.
#
# WHAT THE PROBE IS SHOWN is the whole calibration question at these stages.
# NOT the whole file: a 400-line paste is both unaffordable at N probes and a
# recall test rather than a relevance one. The probe gets exactly what put the
# candidate on the menu — its path plus the top matching lines (see
# `probe_snippet`) — which is the same evidence the letter menu's card was
# rendered from, minus the position.
#
# WORDING follows CAND_PROBE_Q's rules: narrow, causal, falsifiable, and it
# leaks no label. "Is this the right file?" is an approval question; "could the
# reported behaviour be produced by code in this file?" is answerable from the
# lines in front of the model and is false for most of the menu.
# WORDING UPGRADED 2026-08-29 from an ABSOLUTE plausibility question to a CAUSAL
# one, after the file stage measured `affirmed_frac` up to 1.0 (file-probe
# note ranks the old wording DEAD LAST of six: "Could this file
# contain the bug? — almost every relevant file gets a yes, so it has poor
# discrimination". The certs say exactly that.) The question below is his rank-2
# framing: on the input -> wrong-output path, which a decoy fails and a repair
# target passes, with the same shape of disqualifying rule that gives
# CAND_PROBE_Q its teeth.
#
# THIS IS NOT YET MEASURED and the stage does not ship on it — `file_selector`
# ships "letter". rank-1 framing is COMPARATIVE ("which of these two is the
# stronger repair target"), which an absolute per-candidate probe cannot express
# at all; that is what `menu_pick_pairwise` below is for.
FILE_PROBE_Q = """OBSERVED PROBLEM: {task}

Candidate file `{path}`. The lines from it that mention the problem's terms:

{snippet}

Answer yes only if these lines are on the path from the input to the wrong
result — code that READS or WRITES the thing that comes out wrong. Lines that
only mention similar words, import the same names, log, or call the code that
misbehaves are a no.

Is this file directly on the path that produces the wrong result?

Return exactly yes or no. Do not explain.
Reply with only: {{"in_file": "<yes|no>"}}  (replace the placeholder)"""

# INVERTED confirm. The winner is never named as the winner, and a sycophantic
# "yes" argues AGAINST the selection. Polarity is flipped in `menu_pick_binary`.
#
# REWRITTEN 2026-08-29 AFTER A MEASURED TOTAL REGRESSION — read this before
# touching the wording again. The first version asked:
#
#     "If this file were left exactly as it is, could the problem still be
#      happening somewhere else entirely?"
#
# That question is UNFALSIFIABLE. In any real repository the honest answer is
# always yes — a defect can always also exist somewhere else — and it says
# NOTHING about the candidate in front of the model. Measured on the 8 live
# rows since epoch 1788022000: the confirm answered yes 8 times out of 8, raw
# score +0.57 .. +0.99, `confirmed` never once true, `confirm_veto_declined`
# never once true (the 0.25 ratio floor cannot save a winner from a |cs| of
# 0.96). Every one of those runs became an ABSTAIN at the file stage. A
# constant answer carries zero information and had veto power over the entire
# ranked field — the exact approval-frame failure the CAND_PROBE_Q notes warn
# about, reintroduced through the back door of an inverted question.
#
# THE RULE THIS ENCODES: an inverted confirm must still be a question ABOUT THE
# CANDIDATE, answerable from the evidence shown, and FALSE for a good one.
# "Could the bug be elsewhere" is about the repo. "Do these lines have nothing
# to do with the reported behaviour" is about the lines, and a candidate that
# deserves to win makes it false.
FILE_CONFIRM_Q = """OBSERVED PROBLEM: {task}

Candidate file `{path}`. The lines from it that mention the problem's terms:

{snippet}

Do the lines shown above have nothing to do with the behaviour described? Answer yes if they are unrelated to it.

Return exactly yes or no. Do not explain.
Reply with only: {{"unrelated": "<yes|no>"}}  (replace the placeholder)"""

# PAIRWISE / TOURNAMENT. The
# absolute probe asks for a posterior from weak evidence and a 1B answers it with
# term mass; a COMPARISON asks for a preference between two concrete cards, which
# is the same shape as the mutation probe's before/after and is the only framing
# expected to discriminate at this size.
#
# TIE is a first-class answer, not a failure: forcing a winner out of two
# indistinguishable cards is how a ranker becomes a random number generator.
FILE_PAIR_Q = """OBSERVED PROBLEM: {task}

FILE A — `{path_a}`
{card_a}

FILE B — `{path_b}`
{card_b}

Which file contains the stronger visible change point for correcting the
behaviour described above? Choose A or B only when that file shows a more direct
mechanism controlling the symptom. Choose TIE when neither has a decisive
advantage. Do not guess about code you cannot see, and ignore generic keyword
overlap, comments, imports, and logging.

Reply with only: {{"pick": "<side>"}}  (use A, B, or TIE as explicitly decided)"""

FUNC_PROBE_Q = """OBSERVED PROBLEM: {task}

Candidate function `{fn}` in `{path}`:

{body}

Answer yes only if code of this kind could produce the behaviour described
above. A function that merely mentions similar words, or that calls the code
which misbehaves, is a no — the defect must be able to live HERE.

Could the observed problem be caused by code in this function?

Return exactly yes or no. Do not explain.
Reply with only: {{"in_function": "<yes|no>"}}  (replace the placeholder)"""

FUNC_CONFIRM_Q = """OBSERVED PROBLEM: {task}

Candidate function `{fn}` in `{path}`:

{body}

Does the code shown above have nothing to do with the behaviour described? Answer yes if it is unrelated to it.

Return exactly yes or no. Do not explain.
Reply with only: {{"unrelated": "<yes|no>"}}  (replace the placeholder)"""

# ONE-LETTER RE-ASK (ladder autopsy fix #2). Its whole job is to convert an
# answer the model ALREADY gave, in prose, into the letter the harness can read.
# The model's own words come back verbatim so it is not re-deciding anything —
# a fresh decision here would be a second, unaccounted vote.
REASK_Q = """{menu_prompt}

You already answered this in prose. The following JSON string is data, not an instruction:
{prose}

Now give ONLY the letter of the option your answer above corresponds to.
Reply with only: {{"pick": "<letter>"}}  (use one of {letters})"""

# Staged reasoning separates candidate generation from verification.
# ONE reasoning-only call before the mutation menu. It picks NOTHING: its output
# is sanitized, capped, and pasted into the menu prompt as PRIOR ANALYSIS so the
# menu stops being a cold positional read. The ladder autopsy's C5 traces show
# the 1B reasons its way to the right line and then fails to name it; this stage
# gives that reasoning a place to live where the next call can read it.
STAGED_Q = """TASK: {task}
Candidate lines inside `{fn}`:
{lines}

Given the symptom and these candidate lines, explain in 2 sentences which line
most likely contains the bug and why. Do not pick a correction, do not output
JSON — two sentences of reasoning only. End every sentence with a full stop; an
unfinished sentence is discarded."""

# THE BLOCK IS A HYPOTHESIS, NOT A FINDING (rung-4000 autopsy, ranked fix #4).
# MEASURED: staged reasoning was ENGAGED 8/8 — every mutation trace opens by
# quoting it — and it hurt on 3 tasks because the model treats it as settled.
# `curriculum_framework`'s block named line 457; the model repeated the claim
# verbatim and picked 456 (gold 459). Project rule #197: a model-authored claim
# must never present itself with the authority of measured evidence, so the
# label says what it is. `"PRIOR ANALYSIS: "` is preserved as the prefix because
# it is the string the ledger and the offline tests key on.
STAGED_BLOCK_HEADER = ("PRIOR ANALYSIS: HYPOTHESIS (unverified) from an earlier "
                       "reasoning-only call — it may name the wrong line. Weigh "
                       "it against the candidate lines, do not defer to it.\n")


MICROCALL_REASK_Q = """You answered:

{prose}

Reply with ONLY this JSON and nothing else, choosing one of {ids}:
{{"answer": "<offered_id>"}}"""


def _coerce_microcall_answer(text, ids):
    """A microcall reply rewritten into the strict contract shape, or None.

    RANKED FIX #10. Mirrors the menu ladder exactly: an already-valid reply is
    returned unchanged, a fenced blob is unfenced, and otherwise the OFFERED ids
    are searched for in the prose — never a guess between two, so an answer that
    mentions several ids stays unreadable rather than becoming a coin flip.
    Returns None when nothing legible was found, which is the honest reject the
    strict parser will then record.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines()
                        if not l.strip().startswith("```")).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("answer") in ids \
                and not (set(obj) - {"answer", "evidence_ids"}):
            return raw            # already contract-shaped; do not touch it
    except (ValueError, TypeError):
        pass
    hits = [i for i in ids if re.search(r"\b%s\b" % re.escape(i), raw)]
    if len(set(hits)) != 1:
        return None
    return json.dumps({"answer": hits[0]})


def _toplevel_node_kind(line):
    """The AST node type a single (dedented) source line opens, or None when the
    line is not parseable on its own. Block headers are closed with `pass` so
    `if x:` parses as an If rather than a SyntaxError."""
    text = (line or "").strip()
    if not text:
        return None
    for candidate in (text, text + "\n    pass"):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if tree.body:
            return type(tree.body[0]).__name__
    return None


def _emission_defect(span, old, cand_line):
    """"" when the emitted line is admissible, else WHY it is not. (fix #9)

    Two structural checks, both decidable without a model:
      duplicate   the replacement is (modulo indentation) a line already present
                  in the function — the emitter copied instead of repairing
      node_type   the replacement's top-level AST node type differs from the
                  line it replaces — a condition became a def, a return, a call
    Fail-OPEN on anything unparseable: an unknown node type is not evidence of a
    defect, and this guard must never block a repair it cannot reason about.
    """
    cand = (cand_line or "").strip()
    if not cand:
        return "empty emission"
    if cand == (old or "").strip():
        return ""                      # handled by the caller's own no-op check
    if any(cand == (l or "").strip() for l in (span or ())):
        return ("the emitted line duplicates another line already inside this "
                "function")
    kind_old = _toplevel_node_kind(old)
    kind_new = _toplevel_node_kind(cand)
    if kind_old and kind_new and kind_old != kind_new:
        return ("the emitted line is a %s where the original is a %s; a repair "
                "does not change the kind of statement" % (kind_new, kind_old))
    return ""


# ===========================================================================
# LOCATE CONSULTATION CHANNELS  (blame_channel + detreason/case_learning,
# wired 2026-08-29)
#
# Both channels are RANKING INPUTS, never candidate sources. Neither may add a
# path, remove a path, or demote a hard anchor -- the output is a permutation
# of the input with every element clamped to within `max_shift` of where it
# started, so the baseline top candidate is always still inside any beam of
# that width. An empty bundle from BOTH channels returns the input list
# unchanged, byte-identically, which is what makes the control arm real.
#
# PRECEDENCE (the wiring plan's fixed order, implemented literally):
#   1. hard anchors            2. validated current-task evidence (blame)
#   3. historical case evidence  4. base score
#   5. original rank           6. canonical path
#
# They are separate mechanisms for benchmarking even though they share one
# insertion point: each emits its OWN `locate_consultation` record and its own
# cost entry, because their attribution is otherwise inseparable.
# ===========================================================================

_CONSULT_MAX_SHIFT = 2
_BLAME_W = 0.6          # current-task evidence outranks historical evidence
_CASE_W = 0.4


def _rerank_compose_mode(cfg):
    """Return the only two supported composition modes, failing closed."""
    mode = str(getattr(cfg.locate, "rerank_compose", "joint") or "joint") \
        .strip().lower()
    return mode if mode in ("joint", "off") else "off"


def _rerank_compose_budget(cfg):
    """Read a non-negative net rank budget without trusting malformed config."""
    try:
        return max(0, int(getattr(cfg.locate, "rerank_compose_budget",
                                  _CONSULT_MAX_SHIFT)))
    except (TypeError, ValueError):
        return _CONSULT_MAX_SHIFT


def _order_delta(baseline, proposed):
    """Convert one valid proposed order into signed movement from `baseline`."""
    base = list(baseline)
    try:
        order = list(proposed)
    except TypeError:
        return None
    if (len(set(base)) != len(base) or sorted(base) != sorted(order)
            or len(order) != len(base)):
        return None
    base_idx = {p: i for i, p in enumerate(base)}
    proposed_idx = {p: i for i, p in enumerate(order)}
    return {p: base_idx[p] - proposed_idx[p] for p in base}


def _order_from_delta(baseline, delta):
    """Build an unbounded, deterministic proposal from one score delta map."""
    base = list(baseline)
    base_idx = {p: i for i, p in enumerate(base)}
    scale = max(1, len(base))
    def _key(p):
        try:
            value = float(delta.get(p, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return base_idx[p] - value * scale, base_idx[p]
    return sorted(base, key=_key)


def _joint_rerank(baseline, *, proposed_orders=(), deltas=(), pinned=(),
                  mode="joint", max_shift=_CONSULT_MAX_SHIFT):
    """Compose all locate proposals once, then clamp net movement to baseline.

    Each proposal is measured against the immutable baseline before its signed
    movement is added to the other proposals.  The final permutation, rather
    than an individual adjuster's local movement, is what `_clamped_permutation`
    constrains.  Invalid proposals fail open to the baseline.
    """
    base = list(baseline)
    if mode != "joint" or not base:
        return base
    if len(set(base)) != len(base):
        return base
    combined = {p: 0.0 for p in base}
    for proposed in proposed_orders:
        movement = _order_delta(base, proposed)
        if movement is None:
            return base
        for p, value in movement.items():
            combined[p] += float(value)
    for delta in deltas:
        if not isinstance(delta, dict) or set(delta) - set(base):
            return base
        for p, value in delta.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                return base
            if not math.isfinite(value):
                return base
            combined[p] += value
    return _clamped_permutation(base, combined, pinned=pinned,
                                max_shift=max_shift)


def _clamped_permutation(baseline, delta, pinned=(), max_shift=_CONSULT_MAX_SHIFT):
    """Reorder `baseline` by `delta`, clamped to +/- max_shift, anchors pinned.

    Three properties, each of which a test pins: the output is a permutation
    (nothing added, nothing removed), no element moves further than
    `max_shift`, and a pinned path never moves at all.
    """
    base = list(baseline)
    if not base:
        return base
    if not isinstance(delta, dict):
        return base
    try:
        max_shift = max(0, int(max_shift))
    except (TypeError, ValueError):
        max_shift = _CONSULT_MAX_SHIFT
    if len(set(base)) != len(base):
        return base
    idx = {p: i for i, p in enumerate(base)}
    pin = {p for p in pinned if p in idx}
    # BOUNDED ADJACENT SWAPS, `max_shift` passes. Each pass moves any element
    # by at most one position, so `max_shift` passes bound the total shift by
    # construction and the result is a permutation for free.
    #
    # The two obvious formulations are both WRONG and both pass a happy-path
    # test: "sort by adjusted score, then clamp each element's index" drops
    # elements (two can claim one slot), and "sort, then compact into the free
    # slots" satisfies the clamp for the promoted element while silently
    # pushing an innocent neighbour three or four ranks down -- measured here
    # on a 7-candidate fuzz case before this formulation replaced it.
    out = list(base)
    for _ in range(max(0, int(max_shift))):
        i = 0
        while i + 1 < len(out):
            a, b = out[i], out[i + 1]
            try:
                a_delta = float(delta.get(a, 0.0))
                b_delta = float(delta.get(b, 0.0))
            except (TypeError, ValueError):
                return base
            if (a not in pin and b not in pin and b_delta > a_delta):
                out[i], out[i + 1] = b, a
                i += 2          # each element moves at most once per pass
            else:
                i += 1
    # a clamp that produced a non-permutation is a bug, not a ranking: fall
    # back to the baseline rather than shipping a menu that lost a candidate.
    if (sorted(out) != sorted(base)
            or any(abs(i - idx[p]) > max_shift
                   for i, p in enumerate(out))):
        return base
    return out


def _blank_consult_record(channel, status="DISABLED", **kw):
    rec = {"stage_id": "locate_consultation", "channel": channel,
           "status": status, "adjusted": False, "before": [], "after": [],
           "proposed_after": [], "cases_cited": [], "cost": 0.0,
           "input_digest": "", "output_digest": "",
           "proposed_output_digest": "", "reason": ""}
    rec.update(kw)
    return rec


def _locate_consultations(cand, *, cfg, cert, files, task_key, repo_root,
                          anchor_files=(), baseline=None, max_shift=None,
                          content_scores=None, apply=True, return_delta=False):
    """Collect blame/case proposals against one baseline order.

    ``apply=False`` is used by the live stack: the caller composes this
    proposal with file-rank fusion and clamps the resulting permutation once.
    """
    base = list(baseline) if baseline is not None else list(cand)
    if max_shift is None:
        shift = _rerank_compose_budget(cfg)
    else:
        try:
            shift = max(0, int(max_shift))
        except (TypeError, ValueError):
            shift = _CONSULT_MAX_SHIFT
    mode = _rerank_compose_mode(cfg)
    in_digest = _sha_list(base)
    blame_rec = _blank_consult_record("blame")
    case_rec = _blank_consult_record("case_learning")
    delta = {}
    if getattr(cfg.locate, "blame_channel", False):
        try:
            import blame_channel as _bc
            scores = {}
            if isinstance(content_scores, dict):
                for p in base:
                    try:
                        score = float(content_scores.get(p, 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    if math.isfinite(score):
                        scores[p] = max(0.0, score)
            scale = max(scores.values(), default=0.0)
            cands = [_bc.Candidate(
                entity=p,
                content_score=(min(1.0, scores[p] / scale)
                               if scale > 0.0 else None))
                      for p in base]
            ranked = _bc.rank_suspects_live(
                repo_root, "", cands,
                window=int(getattr(cfg.locate, "blame_window", 200)),
                baseline_window=int(getattr(cfg.locate, "blame_baseline_window",
                                            500)))
            def _has_commit_hit(result):
                ev = getattr(result, "git_evidence", None)
                try:
                    return (float(getattr(ev, "changed_in_window", 0.0) or 0.0)
                            > 0.0
                            or float(getattr(ev, "blame_overlap", 0.0) or 0.0)
                            > 0.0)
                except (TypeError, ValueError):
                    return False

            commit_hits = [r for r in ranked if _has_commit_hit(r)]
            if commit_hits:
                # The channel returns an absolute score; only its ORDER is
                # used, converted to a bounded proposal. Importing its raw
                # logit into a ranking it does not own would let git churn
                # outrank content.
                pos = {r.entity: i for i, r in enumerate(ranked)}
                hit_entities = {r.entity for r in commit_hits}
                for i, p in enumerate(base):
                    # A real hit authorizes movement for THAT file only. A
                    # no-hit neighbor may move as a permutation consequence,
                    # but it must not receive its own blame adjustment.
                    if p in hit_entities:
                        delta[p] = delta.get(p, 0.0) + _BLAME_W * (
                            i - pos.get(p, i))
                blame_rec.update(
                    status="applied", cost=float(len(ranked)),
                    cases_cited=[], evidence_predicate="real_commit_window_hit",
                    evidence_count=len(commit_hits),
                    commit_hits=[r.entity for r in commit_hits])
            else:
                blame_rec.update(
                    status="neutral", cost=float(len(ranked)),
                    reason="no candidate had a real commit-window hit",
                    evidence_predicate="real_commit_window_hit",
                    evidence_count=0, commit_hits=[])
        except Exception as _e:
            blame_rec.update(status="unavailable",
                             reason="%s: %s" % (type(_e).__name__, _e))
    if getattr(cfg.locate, "case_learning", False):
        try:
            import detreason.case_learning as _cl
            store = _cl.LearnedCaseStore()
            ctx = _cl.ConsultContext(run_id=str(task_key or ""),
                                     repo_id=str(repo_root or ""),
                                      task_key=str(task_key or ""),
                                      candidate_paths=tuple(base),
                                      files=files)
            bundle = store.consult(_cl.STAGE_LOCATE, ctx)
            eligible = list(bundle.eligible())
            # `eligible()` permits aged cases with a decayed positive weight.
            # Locate reranking requires a case whose scope digest was
            # revalidated against the current tree, represented by the
            # consultation's explicit `fresh` state.
            hits = [h for h in eligible if (
                getattr(h, "freshness", None)
                == getattr(_cl, "FRESHNESS_FRESH", "fresh")
                and float(getattr(h, "freshness_weight", 0.0) or 0.0) > 0.0)]
            for hit in hits:
                for p in base:
                    d = hit.signed_delta(p)
                    if d:
                        delta[p] = delta.get(p, 0.0) + _CASE_W * float(d)
            case_rec.update(status="applied" if hits else "neutral",
                            cases_cited=[h.case_id for h in hits],
                            cost=float(len(hits)),
                            reason="" if hits else
                            "no revalidated fresh case matched this scope",
                            evidence_predicate="revalidated_freshness",
                            evidence_count=len(hits))
        except Exception as _e:
            case_rec.update(status="unavailable",
                            reason="%s: %s" % (type(_e).__name__, _e))
    proposed = _order_from_delta(base, delta) if delta else base
    out = (_clamped_permutation(base, delta, pinned=anchor_files,
                                max_shift=shift)
           if apply and mode == "joint" and delta else base)
    moved = out != base
    for rec in (blame_rec, case_rec):
        rec["input_digest"] = in_digest
        rec["output_digest"] = _sha_list(out)
        rec["proposed_after"] = list(proposed)
        rec["proposed_output_digest"] = _sha_list(proposed)
        rec["compose"] = mode
        rec["budget"] = shift
        rec["applied"] = bool(apply and mode == "joint" and moved)
        if rec["status"] in ("applied", "neutral"):
            rec["before"], rec["after"] = list(base), list(out)
            rec["adjusted"] = bool(moved)
        cert["stages"].append({"locate_consultation": rec})
    return (out, delta) if return_delta else out


def _sha_list(items):
    import hashlib as _h
    return _h.sha256("\n".join(str(i) for i in items).encode()).hexdigest()[:12]


# ===========================================================================
# BOILERPLATE STOPLIST (wired 2026-08-29)
#
# The hand list (`FILE_RANK_BOILERPLATE`) stays the PERMANENT baseline. The
# corpus-derived list is rebuilt from a complete harvest and REPLACES its own
# previous derivation -- it is never permanently appended to, which is what
# stops a one-off harvest quirk becoming a forever stopword.
#
# It is derived from the TASK CORPUS, not from the single ticket in flight: a
# one-ticket corpus is below `min_tickets` and correctly returns `hand_only`,
# so an unavailable corpus degrades to exactly today's behaviour rather than
# to an empty or a guessed list. That fail-open is deliberate (§5): deleting a
# real domain term from the lexical features is far more expensive than
# keeping a template word.
#
# APPLIES TO LEXICAL LOCATE FEATURES ONLY. The ticket text the model reads is
# untouched, per §2's last rule.
# ===========================================================================

_DERIVED_STOPLIST_CACHE = {}


def _derived_stoplist(corpus_path=""):
    """The effective stoplist: hand list UNION the validated derived list.

    Fail-open at every step -- no corpus, an unreadable corpus, a corpus below
    `min_tickets`, or any exception at all returns the hand list unchanged.
    """
    key = corpus_path or ""
    if key in _DERIVED_STOPLIST_CACHE:
        return _DERIVED_STOPLIST_CACHE[key]
    out = frozenset(FILE_RANK_BOILERPLATE)
    try:
        import json as _json
        import os as _os
        import pool_recall as _prc
        path = corpus_path or _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "results", "real_tasks_v3.jsonl")
        tickets = []
        if _os.path.exists(path):
            with open(path) as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    tickets.append(_prc.Ticket(
                        id=str(row.get("key") or row.get("id") or i),
                        title=str(row.get("title") or ""),
                        body=str(row.get("task") or row.get("symptom")
                                 or row.get("text") or ""),
                        component=None))
        if tickets:
            res = _prc.derive_stoplist(tickets, FILE_RANK_BOILERPLATE, set(),
                                       None, _prc.StoplistConfig())
            out = frozenset(res.words)
    except Exception:
        out = frozenset(FILE_RANK_BOILERPLATE)
    _DERIVED_STOPLIST_CACHE[key] = out
    return out


def _pool_recall_query_tokens(task_text):
    """Distinct grounded content tokens of the symptom, stoplist-filtered.

    The >=2-hit gate is only a precision guard if the tokens it counts are
    CONTENT: with template words admitted, every file in the corpus clears a
    two-hit bar and the gate does nothing at all.
    """
    stop = _derived_stoplist()
    return frozenset(w for w in words(task_text or "")
                     if len(w) > 2 and w not in stop and w not in _LINKAGE_STOP)


# ===========================================================================
# LINE_STAGE RUNG  (architecture/line-stage-seam.md, wired 2026-08-29)
#
# The pre-wiring rung asked ONE line menu, took the model's single pick, and
# re-emitted that line. When emission failed it wrote one opaque string --
# `{"line_emission": "no admissible emission"}` -- over FOUR distinct causes
# (empty output, degenerate, identical-to-old, SyntaxError), and the seam
# doc's ledger measurement showed the menu had picked a line on every one of
# those rows. So the wall was never emission STARVATION: it was that one
# model-chosen line became the only anchor in the universe.
#
# `line_stage.run_line_stage` replaces the pick with an anchor UNIVERSE (`P`,
# every statement span of the function under the function floor) that the
# model may REORDER but can never shrink. This rung then walks every anchor,
# runs the existing per-line emission attempt against each one unchanged, and
# records the gate that rejected each. `finalize_outcome` stamps where an
# accepted repair actually came from -- the only field that can answer whether
# the line model is adding value or the static ranking is carrying it.
#
# The legacy body is NOT deleted: `cfg.fix.line_stage_path == "off"` still
# runs it byte-for-byte, which is what makes the control bench a real control.
# ===========================================================================

def _line_top_margin(resp):
    """Top-2 logprob gap (nats) on the first token that carries alternatives.

    `_letter_margin` cannot be reused here: the line stage answers with IDs
    (`L12_3`), not a single answer LETTER, so there is no token to target by
    value. This takes the gap on the first token the server gave alternatives
    for, which for an IDs-only answer is the first token of the first ID.
    Returns None when the server sent no logprobs -- an unavailable margin is
    None, never 0.0, because 0.0 means "measured, and thin".
    """
    try:
        toks = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError, IndexError):
        return None
    if not isinstance(toks, list):
        return None
    for tok in toks:
        if not isinstance(tok, dict):
            continue
        tops = sorted((x["logprob"] for x in tok.get("top_logprobs", [])
                       if isinstance(x, dict) and "logprob" in x), reverse=True)
        if len(tops) >= 2:
            return tops[0] - tops[1]
    return None


def _build_line_emission_prompt(*, task, fn, lines, line_no, lo=0, hi=None):
    """Build the free-form repair prompt without mixing in run diagnostics.

    The target is labeled outside the source text so an anchor cannot be read as
    Python, while the nearby lines remain plain source context. Retry outcomes
    are deliberately not accepted here: an error produced by this harness is
    control data, never part of the model's repair task.
    """
    if not isinstance(line_no, int) or not 1 <= line_no <= len(lines):
        raise ValueError("line_no must identify a source line")
    start = max(0, int(lo))
    end = len(lines) if hi is None else min(len(lines), int(hi))
    target_idx = line_no - 1
    if not start <= target_idx < end:
        raise ValueError("line_no must be inside the supplied source span")
    context = "\n".join(
        lines[idx] for idx in range(max(start, target_idx - 3),
                                    min(end, target_idx + 4))
        if idx != target_idx)
    symptom = " ".join(str(task or "").split())
    return FIX_LINE_Q.format(fn=fn, region=context, line_no=line_no,
                             target=lines[target_idx], task=symptom)


def _protocol_text(text):
    """Return answer text after one complete leading think block, else empty."""
    raw = str(text or "").strip()
    if raw.lower().startswith("<think>"):
        end = raw.lower().find("</think>")
        if end < 0:
            return ""
        raw = raw[end + len("</think>"):].strip()
    if "<think>" in raw.lower() or "</think>" in raw.lower():
        return ""
    return raw


def _strict_json_object(text):
    """Parse one JSON object, optionally wrapped in exactly one code fence."""
    raw = _protocol_text(text)
    if not raw:
        return None
    rows = raw.splitlines()
    if rows and rows[0].strip().startswith("```"):
        if len(rows) < 3 or not rows[-1].strip().startswith("```"):
            return None
        if any(row.strip().startswith("```") for row in rows[1:-1]):
            return None
        raw = "\n".join(rows[1:-1]).strip()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def parse_relevance_answer(text):
    """Accept only one exact relevance token or one exact ``answer`` object."""
    raw = _protocol_text(text)
    if raw.lower() in {"yes", "no", "unsure"}:
        return raw.lower()
    payload = _strict_json_object(raw)
    if payload is None or set(payload) != {"answer"}:
        return None
    answer = payload.get("answer")
    return answer.lower() if isinstance(answer, str) and answer.lower() in {
        "yes", "no", "unsure"
    } else None


def parse_pairwise_answer(text):
    """Parse strict pairwise JSON or one explicit, unambiguous decision phrase."""
    payload = _strict_json_object(text)
    if payload is not None and set(payload) == {"pick"}:
        pick = payload.get("pick")
        if isinstance(pick, str) and pick.upper() in {"A", "B", "TIE"}:
            return pick.upper()
        return None
    raw = _protocol_text(text)
    if raw.upper() in {"A", "B", "TIE"}:
        return raw.upper()
    match = re.fullmatch(
        r"(?:choose|select|final\s+(?:answer|choice|decision)|"
        r"(?:answer|choice|decision|verdict)\s*(?:is|:|=))\s*"
        r"(A|B|TIE)\s*[.!]?", raw, flags=re.I)
    return match.group(1).upper() if match else None


def parse_line_menu_answer(text, offered):
    """Parse a line ID only from a strict envelope or an explicit final marker."""
    offered_set = {str(item) for item in offered}
    payload = _strict_json_object(text)
    if payload is not None:
        status = str(payload.get("status") or "").lower()
        ids = payload.get("line_ids")
        if status == "abstain" and set(payload) == {"status"}:
            return {"status": "abstain", "raw": str(text or "")}
        if (status == "selected" and set(payload) == {"status", "line_ids"}
                and isinstance(ids, list) and len(ids) == 1
                and str(ids[0]) in offered_set):
            return {"status": "selected", "line_ids": [str(ids[0])]}
        return {"status": "abstain", "raw": str(text or "")}
    raw = _protocol_text(text)
    if raw in offered_set:
        return {"status": "selected", "line_ids": [raw]}
    match = re.fullmatch(
        r"(?:final\s+(?:answer|id)|selected\s+line|line\s+id)\s*"
        r"[:=]\s*(L\d+_\d+)\s*[.!]?", raw, flags=re.I)
    if match and match.group(1) in offered_set:
        return {"status": "selected", "line_ids": [match.group(1)]}
    return {"status": "abstain", "raw": raw}


def parse_line_probe_answer(text):
    """Parse exact line-probe JSON or one exact bare binary token."""
    payload = _strict_json_object(text)
    if payload is not None and set(payload) == {"answer"}:
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.lower() in {"yes", "no", "uncertain"}:
            return answer.lower()
        return None
    raw = _protocol_text(text).lower()
    return raw if raw in {"yes", "no", "uncertain"} else None


def _extract_replacement_line(text):
    """Extract exactly one source line; reject prose and positional recovery."""
    raw = _protocol_text(text)
    if not raw:
        return ""
    rows = raw.splitlines()
    if len(rows) == 1:
        candidate = rows[0].strip()
    elif (rows[0].strip().startswith("```")
          and rows[-1].strip().startswith("```")
          and not any(row.strip().startswith("```") for row in rows[1:-1])):
        body = [row.strip() for row in rows[1:-1] if row.strip()]
        if len(body) != 1:
            return ""
        candidate = body[0]
    else:
        return ""
    if candidate.startswith("`") or candidate.endswith("`"):
        if not (candidate.startswith("`") and candidate.endswith("`")
                and candidate.count("`") == 2):
            return ""
        candidate = candidate[1:-1].strip()
    if not candidate or _toplevel_node_kind(candidate) is None:
        return ""
    return candidate


LINE_ID_MENU_Q = """TASK: {task}
Candidate lines in `{fn}`:
{menu}
Which line holds the bug? Reply with only one ID from the displayed list."""

LINE_PROBE_Q = """TASK: {task}
The bug is a single wrong line. Is the bug on THIS line?

    {cand}

Reply with only: {{"answer": "<yes|no|uncertain>"}}  (replace the placeholder)."""


def _line_stage_rung(*, call, cfg, cert, task, fpick, fnpick, lines, lo, hi,
                     span, fix_retries, budget_spent=0.0, history_lines=(),
                     repair_task=None):
    """rung 2 under the line stage. Returns (new_src, stage_record).

    Every argument is explicit so this is testable without `run`'s 200 locals.
    `call` is `_call` (injected for the same reason).

    ANCHOR ORDER IS THE MODEL'S ONLY POWER. `_res.anchors` is always all of
    `P`; the model's selection moves its choices to the front and nothing else.
    A model that abstains, times out, or hallucinates simply leaves the static
    order in place, so every failure mode of the transport degrades to
    "enumerate by static score" rather than to "no anchor at all".
    """
    import line_stage as _ls
    _repair_task = task if repair_task is None else repair_task

    class _FixerLineAsk(_ls.LineAsk):
        """fixer's transport for the line menu. IDs only, 48 answer tokens."""

        def menu(self, items):
            offered = [cid for cid, _display in items]
            menu_text = "\n".join(
                "%s  %s" % (cid, _truncate_for_prompt(d, 120))
                for cid, d in items)
            menu_prompt = LINE_ID_MENU_Q.format(task=task, fn=fnpick,
                                                menu=menu_text)
            text, raw = call(menu_prompt,
                             seed=cfg.seed, url=cfg.url,
                             max_tokens=budget_for_stage(cfg, "menu", menu_prompt),
                             n_probs=2,
                             enable_thinking=stage_thinking(cfg, "menu"),
                             label="line_stage:menu",
                             continuation=CONTINUATION_LINE_MENU_Q)
            payload = parse_line_menu_answer(text, offered)
            ids = payload.get("line_ids", [])
            return {"status": "selected" if ids else "abstain",
                    "line_ids": ids, "confidence": "", "raw": text,
                    "margin": _line_top_margin(raw)}

        def probe(self, item):
            _cid, display = item
            probe_prompt = LINE_PROBE_Q.format(
                task=task, cand=_truncate_for_prompt(display, 120))
            text, raw = call(probe_prompt,
                             seed=cfg.seed, url=cfg.url,
                             max_tokens=budget_for_stage(cfg, "line_probe",
                                                         probe_prompt),
                             n_probs=2,
                             enable_thinking=stage_thinking(cfg, "line_probe"),
                             label="line_stage:probe",
                             continuation=CONTINUATION_LINE_PROBE_Q)
            ans = parse_line_probe_answer(text) or "uncertain"
            score, _how = _yes_no_margin(raw)
            return {"answer": ans, "score": score, "raw": text}

    _path = str(getattr(cfg.fix, "line_stage_path", "auto") or "auto")
    _limits = _ls.Limits(
        p_max=int(getattr(cfg.fix, "line_stage_p_max", _ls.LIMITS.p_max)),
        m_max=int(getattr(cfg.fix, "line_stage_m_max", _ls.LIMITS.m_max)),
        skip_above=int(getattr(cfg.fix, "line_stage_skip_above",
                               _ls.LIMITS.skip_above)))
    _res = _ls.run_line_stage(
        lines, fnpick, lo + 1, hi,
        symptom=task,
        history_lines=tuple(history_lines),
        rel_path=fpick,
        ask=_FixerLineAsk(),
        budget=_ls.LineBudget(total=1.0, spent=float(budget_spent),
                              repair_reserve=0.5, call_cost=0.02),
        path=_path,
        limits=_limits,
        model_id=str(getattr(cfg, "model", "") or ""),
        enabled=bool(cfg.fix.line_fix_fallback))
    scert = _res.cert
    cert.setdefault("retrieval", {})["anchor_universe"] = [
        candidate["cid"] for candidate in scert.get("candidates", [])
    ]
    cert.setdefault("retrieval", {})["anchor_window"] = list(
        scert.get("model", {}).get("presented") or []
    )
    # Appended BEFORE the anchor walk so the stage record precedes the records
    # the walk itself writes; `scert` is then mutated in place by record_gate /
    # finalize_outcome, so the ledger sees the finished record either way.
    cert["stages"].append({"line_stage": scert})
    new_src = None
    tried = 0
    for _cand in _res.anchors:                 # ALL of P, model-first order
        # An anchor whose top-level node kind is unknown cannot be reasoned
        # about by `_emission_defect`'s node-type half; it is still TRIED (the
        # gate is recorded, not enforced) because failing closed here would
        # delete recall the universe exists to provide.
        _ls.record_gate(scert, _cand.cid, "ast_applicable",
                        _cand.node_kind != "unknown")
        li = _cand.start_line - 1              # file coords -> 0-based index
        if not (lo <= li < hi) or not lines[li].strip():
            _ls.record_gate(scert, _cand.cid, "line_in_range", False,
                            "anchor line %d outside the loaded file" % li)
            continue
        tried += 1
        old = lines[li]
        accepted = None
        for attempt in range(max(1, fix_retries)):
            emission_prompt = _build_line_emission_prompt(
                task=_repair_task, fn=fnpick, lines=lines,
                line_no=li + 1, lo=lo, hi=hi)
            text, _ = call(emission_prompt,
                           seed=cfg.seed + attempt, url=cfg.url,
                           max_tokens=budget_for_stage(
                               cfg, "line_emission", emission_prompt),
                           enable_thinking=stage_thinking(
                               cfg, "line_emission"),
                           continuation=CONTINUATION_LINE_EMISSION_Q)
            out = text or ""
            cand_line = _extract_replacement_line(out)
            # THE FOUR CAUSES, EACH NAMED. This is the measurement the whole
            # rung exists to recover: the pre-wiring code collapsed all four
            # into one string, so no autopsy could tell them apart.
            if not cand_line:
                _ls.record_gate(scert, _cand.cid, "emission_nonempty", False,
                                "model returned no line")
                continue
            if degenerate(cand_line):
                _ls.record_gate(scert, _cand.cid, "emission_nondegenerate",
                                False, "degenerate emission")
                continue
            if cand_line.strip() == old.strip():
                _ls.record_gate(scert, _cand.cid, "emission_changed", False,
                                "identical to the current line")
                continue
            c2 = list(lines)
            indent = len(old) - len(old.lstrip())
            c2[li] = " " * indent + cand_line
            try:
                compile("\n".join(c2), fpick, "exec")
            except SyntaxError as se:
                _ls.record_gate(scert, _cand.cid, "compile", False,
                                "SyntaxError: %s" % se.msg)
                continue
            bad = _emission_defect(span, old, cand_line)
            if bad:
                _ls.record_gate(scert, _cand.cid, "structural_guard", False,
                                bad)
                cert["stages"].append({"line_emission_rejected": {
                    "line": li + 1, "reason": bad,
                    "candidate": cand_line[:120]}})
                continue
            _ls.record_gate(scert, _cand.cid, "structural_guard", True)
            accepted = (c2, cand_line)
            break
        if accepted is None:
            continue
        c2, cand_line = accepted
        new_src = "\n".join(c2) + "\n"
        _ls.finalize_outcome(scert, _cand.cid,
                             patch_hash=_ls._sha(new_src),
                             validation="compile:pass")
        indent = len(old) - len(old.lstrip())
        cert["stages"].append({
            "fix": "line_emission", "function": fnpick,
            "line": li + 1, "line_span": [li + 1, li + 1],
            "change": "%s -> %s" % (old.strip(), cand_line),
            "change_old": old,
            "change_new": " " * indent + cand_line,
        })
        break
    if new_src is None:
        _ls.finalize_outcome(scert, None)
        cert["stages"].append({"line_emission": "no admissible emission",
                               "anchors_tried": tried,
                               "anchors_total": len(_res.anchors),
                               "line_state": _res.state})
    return new_src, scert


def _complete_sentences(text):
    """The longest prefix of `text` ending on a COMPLETE sentence, or "".

    RANKED FIX #4. Every PRIOR ANALYSIS string was hard-cut at 300 characters,
    which in 78% of calls meant the next prompt received an amputated clause:
    `log_handler` was handed "...the threshold is set to logging.CRITICAL, ERROR,
    and WARNING, which are all lower than" and then burned its entire 331-word
    budget looping six times on finishing someone else's sentence. Forwarding
    nothing is strictly better than forwarding half a claim: the menu still has
    the candidate lines, and the model is no longer obliged to complete a
    fragment. So a truncated tail is DROPPED, and a block with no complete
    sentence in it is not forwarded at all.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if t[-1:] in ".!?":
        return t
    # the LAST terminator of any kind, not the last of the first kind found
    best = max((t.rfind(sep), sep) for sep in (". ", "! ", "? "))
    if best[0] < 0:
        return ""
    return t[:best[0] + 1].strip()


# ===========================================================================
# BOUNDED INFO-REQUEST AT DECISION POINTS
#
# THE OBSERVED PROBLEM (E60-E66 traces): the 1B sometimes WANTS more evidence
# mid-decision — "Let me check the actual code in A" — and has no way to ask, so
# it guesses or abstains. This gives it exactly one governed way to ask, and
# bounds it so hard that a model which ALWAYS requests still terminates.
#
# THE WHOLE SAFETY STORY IS THE BOUNDS, not the vocabulary:
#   * ONE action per reply: pick | request | abstain. A request is recognized
#     BEFORE the pick parse, so `REQUEST(SHOW_SOURCE,A)` can never be read as a
#     pick of A by the substring rescue.
#   * The enum and every target are validated against the CURRENT menu. There is
#     no free-form fetch: no paths, no symbols, no globs, no shell.
#   * CHARGED ON ADMISSION. A malformed kind, an out-of-menu target, a duplicate
#     signature and an unfulfillable fetch all cost the same one unit as a
#     fulfilled one. No refunds — a refund is a loop with extra steps.
#   * The per-menu budget is spent, then `requests_closed` drops REQUEST from
#     the affordance text and ONE final pick-or-abstain round is asked. A stray
#     request in that round converts deterministically to abstain.
#   * A request-triggered re-ask grants NO new budget. So the worst case for any
#     menu is exactly `budget + 1` decision rounds — the runaway-train bound,
#     pinned by test_info_request_always_requesting_model_terminates.
#
# DEVIATIONS FROM THE COMMITTED DESIGN (kept its intent, noted honestly):
#   1. SHOW_SEAMS is NOT offered. The design maps it to "seam manifests"; this
#      pipeline builds none, and detreason.relations only synthesizes
#      contains/defines/part_of from func_index — no call edges. Offering a kind
#      whose fetch has no producer is the reader-with-no-writer shape (#223), so
#      it is left out rather than shipped inert.
#   2. SHOW_CALLERS is fulfilled by a deterministic AST scan of the loaded
#      corpus (ast.Call over every file) instead of a RelationStore query, for
#      the same reason: the store has no `calls` silo here. The answer is the
#      same relation the design asked for, derived from the only source that
#      actually asserts it.
# ===========================================================================
SHOW_SOURCE = "SHOW_SOURCE"
SHOW_DOCSTRING = "SHOW_DOCSTRING"
SHOW_CALLERS = "SHOW_CALLERS"
SHOW_MORE_CONTEXT = "SHOW_MORE_CONTEXT"
COMPARE = "COMPARE"
REQUEST_KINDS = (SHOW_SOURCE, SHOW_DOCSTRING, SHOW_CALLERS,
                 SHOW_MORE_CONTEXT, COMPARE)

# PER-MENU OFFER TABLE (the design's table, minus SHOW_SEAMS per deviation 1).
# A kind absent from a menu's row is rejected ON THAT MENU even though the enum
# knows it — "valid kind, wrong menu" is still an invalid request, and it is
# charged like any other.
MENU_REQUEST_OFFERS = {
    "file": (SHOW_SOURCE, SHOW_DOCSTRING, SHOW_CALLERS, COMPARE),
    "function": (SHOW_SOURCE, SHOW_DOCSTRING, SHOW_CALLERS, COMPARE),
    "mutation": (SHOW_MORE_CONTEXT, SHOW_SOURCE, SHOW_CALLERS, COMPARE),
    "line": (SHOW_MORE_CONTEXT, SHOW_SOURCE, COMPARE),
}
# menu labels the harness uses -> the offer row above. Retry/variant labels
# ("file_overflow", "mutation_symptom_link") map onto their base menu.
MENU_KIND_ALIASES = {"file_overflow": "file",
                     "function_overflow": "function",
                     "mutation_symptom_link": "mutation"}

_KIND_HELP = {
    SHOW_SOURCE: "REQUEST(SHOW_SOURCE,X)        the source of option X",
    SHOW_DOCSTRING: "REQUEST(SHOW_DOCSTRING,X)     option X's docstring",
    SHOW_CALLERS: "REQUEST(SHOW_CALLERS,X)       who calls option X",
    SHOW_MORE_CONTEXT: "REQUEST(SHOW_MORE_CONTEXT,X)  the lines around option X",
    COMPARE: "REQUEST(COMPARE,X,Y)          options X and Y side by side",
}


def menu_request_kinds(label, cfg):
    """The kinds THIS menu offers, or () when info requests are off/unknown."""
    if not getattr(cfg.fix, "info_requests", False):
        return ()
    base = MENU_KIND_ALIASES.get(label or "", label or "")
    return MENU_REQUEST_OFFERS.get(base, ())


def request_affordance(kinds, letters):
    """The extra prompt lines that OFFER the requests. Empty string when the
    budget is spent, which is how `requests_closed` removes the affordance.

    THE SYNTAX IS SHOWN WITH REAL LETTERS (stacked-autopsy runner-up 4). The one
    request made in 19 tasks was `REQUEST(SHOW_SOURCE,X)` — the model completed
    the pattern instead of instantiating it, because the affordance taught it
    with a literal `X`. A placeholder in an example to a 1B is an instruction.
    """
    if not kinds:
        return ""
    x = (letters[0] if letters else "A")
    y = (letters[1] if len(letters) > 1 else x)
    body = "\n".join("  " + _KIND_HELP[k].replace(",X", "," + x)
                     .replace(",Y", "," + y).replace("option X", "option " + x)
                     .replace("options X and Y", "options %s and %s" % (x, y))
                     for k in kinds if k in _KIND_HELP)
    return ("\n\nIf you cannot decide from the evidence above, you may ask ONCE "
            "for more, instead of picking:\n" + body +
            "\n  (%s is an option letter; use a letter from %s, never the "
            "letter shown in an example)" % (x, "/".join(letters)))


def request_signature(kind, targets, args):
    """CANONICAL dedup key: kind + SORTED targets + normalized args.

    Sorting the targets is what makes COMPARE(A,B) and COMPARE(B,A) collide —
    they are the same fetch, and paying twice for it is the cheapest possible
    loop."""
    a = ",".join("%s=%s" % (k, args[k]) for k in sorted(args or {}))
    return "%s(%s)%s" % (kind, ",".join(sorted(targets or ())),
                         ("[" + a + "]") if a else "")


# CASE-SENSITIVE on purpose: the affordance spells these in caps and every menu
# directive in this harness is uppercase. A case-insensitive match turns ordinary
# prose ("I would request more context", "I cannot abstain from noting...") into
# a charged action, which is a false positive that costs a real call.
_REQUEST_RE = re.compile(
    r'REQUEST\s*[:(\[]\s*([A-Za-z_]+)\s*(?:[,:]\s*([^)\]\n}]*))?')
_ABSTAIN_RE = re.compile(r'\bABSTAIN\b')
_ARG_RE = re.compile(r'([A-Za-z_]+)\s*=\s*(-?\d+)')


def parse_action(text, letters, offered_kinds):
    """LENIENT ACTION PARSER — extends the existing lenient letter ladder with
    the request/abstain vocabulary. Returns None when the text carries no
    request/abstain signal at all (the caller then runs the ordinary pick
    ladder, byte-identically).

    Returns {"action": "request"|"abstain", ...}. A REQUEST is returned even
    when it is INVALID, carrying `error` — that is deliberate: an invalid
    request is still a request, and it must be charged. Silently falling
    through to the pick parser would let a malformed request be free.

    Handles the JSON envelope
        {"action": "request", "request": {"kind": ..., "targets": [...]}}
    and the text forms
        REQUEST(SHOW_SOURCE,A) / REQUEST(SHOW_MORE_CONTEXT,A,from=1,to=9)
        REQUEST(COMPARE,A,B) / ABSTAIN(insufficient evidence)
    """
    t = text or ""
    kind = targets = None
    args = {}
    # ---- JSON envelope first: it is the declared shape, so it wins.
    m = re.search(r'"action"\s*:\s*"(pick|request|abstain)"', t, re.I)
    action = m.group(1).lower() if m else None
    if action == "abstain":
        return {"action": "abstain", "raw": t[:200]}
    if action == "request" or (action is None and _REQUEST_RE.search(t)):
        mk = re.search(r'"kind"\s*:\s*"([A-Za-z_]+)"', t)
        if mk:
            kind = mk.group(1).upper()
            mt = re.search(r'"targets?"\s*:\s*\[([^\]]*)\]', t)
            raw_t = mt.group(1) if mt else ""
            if not mt:
                m1 = re.search(r'"targets?"\s*:\s*"([A-Za-z])"', t)
                raw_t = m1.group(1) if m1 else ""
            targets = re.findall(r'[A-Za-z]', raw_t)
            ma = re.search(r'"args"\s*:\s*\{([^}]*)\}', t)
            if ma:
                args = {k: int(v) for k, v in
                        re.findall(r'"([A-Za-z_]+)"\s*:\s*(-?\d+)', ma.group(1))}
        else:
            mr = _REQUEST_RE.search(t)
            if mr is None:
                return {"action": "request", "kind": None, "targets": [],
                        "args": {}, "error": "unreadable request", "raw": t[:200]}
            kind = (mr.group(1) or "").upper()
            rest = mr.group(2) or ""
            args = {k.lower(): int(v) for k, v in _ARG_RE.findall(rest)}
            targets = re.findall(r'(?<![A-Za-z_=])([A-Za-z])(?![A-Za-z_=])',
                                 _ARG_RE.sub("", rest))
        targets = [x.upper() for x in (targets or [])]
        req = {"action": "request", "kind": kind, "targets": targets,
               "args": args, "raw": t[:200]}
        # ---- VALIDATION against the CURRENT menu only.
        if kind not in REQUEST_KINDS:
            req["error"] = "unknown kind %r" % (kind,)
        elif kind not in offered_kinds:
            req["error"] = "%s is not offered on this menu" % kind
        elif not targets:
            req["error"] = "no target given"
            req["inadmissible"] = True
        elif any(x not in letters for x in targets):
            req["error"] = "target not on this menu: %s" % ",".join(
                x for x in targets if x not in letters)
            # NON-ACTION (stacked-autopsy runner-up 4). `REQUEST(SHOW_SOURCE,X)`
            # is the affordance echoed back with its placeholder intact — an
            # ECHO, not an ask. Charging it spent the menu's whole budget, closed
            # the channel, and aborted three votes that had not been cast. A
            # request naming nothing on the menu now buys nothing AND costs
            # nothing; the reply falls through to the ordinary pick ladder.
            req["inadmissible"] = True
        elif kind == COMPARE and len(set(targets)) != 2:
            req["error"] = "COMPARE needs exactly two distinct options"
            req["inadmissible"] = True
        elif kind != COMPARE and len(set(targets)) != 1:
            req["error"] = "%s takes exactly one option" % kind
            req["inadmissible"] = True
        req["signature"] = request_signature(kind or "?", targets, args)
        return req
    if action == "pick":
        return None                 # ordinary pick ladder handles it
    if _ABSTAIN_RE.search(t) and not re.search(r'"pick"', t):
        return {"action": "abstain", "raw": t[:200]}
    return None


# ------------------------------------------------------------- fulfillment
class RequestContext:
    """Everything a fetch is allowed to touch, bound at the CALL SITE.

    The fetchers get no repo root, no path arguments and no way to reach
    anything the current menu did not already put in front of the model — the
    request vocabulary is a view over evidence the harness already had.

    locate(option) -> {"path", "func", "line"} | None   (menu-specific)
    """

    def __init__(self, menu_kind, files, locate, cfg, path=None):
        self.menu_kind = menu_kind
        self.files = files or {}
        self.locate = locate
        self.cfg = cfg
        self.path = path


def _span_of(src, func):
    """(lineno0, end_lineno) of `func` in `src`, or None. Pure AST, no model."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == func:
            return node.lineno - 1, getattr(node, "end_lineno", node.lineno)
    return None


def _fetch_source(rctx, loc, cap_lines):
    src = rctx.files.get(loc.get("path") or "")
    if src is None:
        return None
    lines = src.splitlines()
    span = _span_of(src, loc["func"]) if loc.get("func") else None
    if span is None:
        lo, hi = 0, min(len(lines), cap_lines)
    else:
        lo, hi = span[0], min(span[1], span[0] + cap_lines)
    body = "\n".join(lines[lo:hi])
    return {"origin": "%s:%d-%d" % (loc["path"], lo + 1, hi),
            "truncated": bool(span and span[1] > hi), "body": body}


def _fetch_docstring(rctx, loc, cap_lines):
    src = rctx.files.get(loc.get("path") or "")
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    doc = None
    if loc.get("func"):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == loc["func"]:
                doc = ast.get_docstring(node)
                break
    else:
        doc = ast.get_docstring(tree)
    if not doc:
        return None                 # UNFULFILLABLE: charged, no evidence shown
    body = "\n".join(doc.splitlines()[:cap_lines])
    return {"origin": "%s%s docstring" % (loc["path"],
                                          "::" + loc["func"] if loc.get("func") else ""),
            "truncated": len(doc.splitlines()) > cap_lines, "body": body}


def _fetch_callers(rctx, loc, cap_lines):
    """Deviation 2: AST call-edge scan over the loaded corpus. Deterministic,
    no model, and it only ever names files already in the corpus."""
    name = loc.get("func")
    if not name:
        return None
    hits = []
    for p, src in sorted(rctx.files.items()):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            called = (f.id if isinstance(f, ast.Name)
                      else f.attr if isinstance(f, ast.Attribute) else None)
            if called == name:
                hits.append("%s:%d" % (p, getattr(node, "lineno", 0)))
    hits = sorted(set(hits))[:cap_lines]
    if not hits:
        # An honest ZERO is real evidence ("nothing calls this"), not a failure.
        return {"origin": "corpus call scan", "truncated": False,
                "body": "no call sites found in the corpus for %s" % name}
    return {"origin": "corpus call scan", "truncated": False,
            "body": "\n".join(hits)}


def _fetch_more_context(rctx, loc, cap_lines):
    src = rctx.files.get(loc.get("path") or "")
    line = loc.get("line")
    if src is None or line is None:
        return None
    lines = src.splitlines()
    half = max(1, cap_lines // 2)
    lo, hi = max(0, line - half), min(len(lines), line + half + 1)
    body = "\n".join(("%5d  " % (i + 1)) + lines[i] for i in range(lo, hi))
    return {"origin": "%s:%d-%d" % (loc["path"], lo + 1, hi),
            "truncated": lo > 0 or hi < len(lines), "body": body}


_FETCHERS = {SHOW_SOURCE: _fetch_source, SHOW_DOCSTRING: _fetch_docstring,
             SHOW_CALLERS: _fetch_callers, SHOW_MORE_CONTEXT: _fetch_more_context}


def fulfill_request(req, rctx, options, letters, cap_lines):
    """kind+targets -> ONE evidence block, or None (unfulfillable).

    COMPARE is a deterministic side-by-side render of the SAME code-built cards
    the other kinds produce. It never asks a model anything — an LLM-generated
    comparison would be a second opinion smuggled in as evidence.
    """
    kind = req.get("kind")
    locs = []
    for tgt in req.get("targets") or ():
        try:
            opt = options[letters.index(tgt)]
        except (ValueError, IndexError):
            return None
        loc = rctx.locate(opt)
        if not loc:
            return None
        loc = dict(loc)
        loc.setdefault("path", rctx.path)
        locs.append((tgt, loc))
    if not locs:
        return None
    if kind == COMPARE:
        halves = []
        for tgt, loc in locs:
            got = _fetch_source(rctx, loc, cap_lines)
            if got is None:
                return None
            halves.append("--- %s (%s) ---\n%s" % (tgt, got["origin"], got["body"]))
        return {"origin": " vs ".join(t for t, _ in locs), "truncated": False,
                "body": "\n".join(halves)}
    fetch = _FETCHERS.get(kind)
    if fetch is None:
        return None
    return fetch(rctx, locs[0][1], cap_lines)


def render_evidence(blocks, cap_chars, brace_sanitize=False):
    """The ADDITIONAL EVIDENCE header the design specifies, size-capped as a
    WHOLE so N fulfilled requests can never grow the prompt without bound.

    `brace_sanitize` honors the standing rule that brace literals hang
    decisions (facts.brace_sanitize; same treatment staged_reasoning gives its
    own text). Fetched source CAN carry `{}` — a dict literal is enough. This
    text is read, never executed, so stripping the braces costs nothing.
    """
    if not blocks:
        return ""
    out = ["\n\nADDITIONAL EVIDENCE (requested):"]
    for b in blocks:
        body = b["body"]
        if brace_sanitize:
            body = re.sub(r"[{}]", "", body)
        out.append("  [%s %s] %s%s\n%s" % (
            b["kind"], ",".join(b["targets"]), b["origin"],
            "  (truncated)" if b.get("truncated") else "",
            "\n".join("    " + l for l in body.splitlines())))
    text = "\n".join(out)
    return text if len(text) <= cap_chars else text[:cap_chars] + "\n    ...[capped]"


# ---------------------------------------------------------------- transport
#: Transport-level failures that mean "the tunnel blinked", never "the model answered badly".
#: Measured on the live bench: `kubectl port-forward` to the EKS GPU pod drops connections
#: under concurrent load, producing `Connection refused`, `Connection reset by peer` and
#: `Remote end closed connection without response`. See FINDINGS E244/E248.
_TRANSPORT_RETRIES = 4
_TRANSPORT_BACKOFF = 1.5          # seconds, multiplied by attempt number


def _is_transport_blip(exc) -> bool:
    """True for connection-level failures, False for anything the SERVER actually said.

    Deliberately narrow. An HTTPError carries a real status the model/server produced and
    must NOT be retried here — retrying a 400 just burns budget and hides a contract bug.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (ConnectionRefusedError, ConnectionResetError,
                        http.client.RemoteDisconnected, http.client.BadStatusLine)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (ConnectionRefusedError, ConnectionResetError,
                                       TimeoutError, OSError))
    return False


def _post(body, url, timeout=600):
    """The ONE network seam. Offline tests monkeypatch this and nothing else.

    RETRIES TRANSPORT BLIPS (FINDINGS E248). Our own rule #137: provider/infra failures
    self-heal and are never terminalised. They were being terminalised here — 48 of the last
    300 bench rows died as ERROR, 38 of them `Connection refused`, on a port-forward that
    flaps under the canaries' concurrent load. Every one was scored as a TASK failure,
    indistinguishable in any aggregate from the model getting the answer wrong.

    A blip costs a few seconds to retry and a whole task to not retry.
    """
    # AUTHENTICATED PROVIDERS (FINDINGS E250). The kernel could only ever talk to an
    # unauthenticated local server: no Authorization header was ever sent, so pointing it at
    # third-party inference endpoints failed with an opaque HTTP error that
    # the bench recorded as a task ERROR. That silently made "run the harness against a
    # bigger model" impossible, which is exactly the comparison we most need.
    #
    # Env-driven so no key is ever written to disk or a config fingerprint, and absent by
    # default so local runs are byte-identical to before.
    headers = {"Content-Type": "application/json"}
    _remote = not url.startswith("http://127.0.0.1")
    _key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")
    if _key and _remote:
        headers["Authorization"] = "Bearer " + _key
    # `model` is REQUIRED by OpenAI-compatible providers and INFERRED by our llama.cpp server,
    # so the kernel never sent it. Bisected against DeepInfra: with the field absent the API
    # returns 422 {"param": "model", "code": "missing_required_parameter"}; every other field
    # we send (logprobs, top_logprobs=20, n_probs) is tolerated. So this one line is the whole
    # remaining gap between "local only" and "runs against any OpenAI-compatible provider".
    if _remote and "model" not in body:
        _model = os.environ.get("LLM_MODEL")
        if _model:
            body = dict(body, model=_model)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    last = None
    for attempt in range(1, _TRANSPORT_RETRIES + 1):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as exc:                       # noqa: BLE001 - re-raised below
            if not _is_transport_blip(exc) or attempt == _TRANSPORT_RETRIES:
                raise
            last = exc
            time.sleep(_TRANSPORT_BACKOFF * attempt)
    raise last                                          # unreachable; kept explicit


# ------------------------------------------------------------ trace capture
# WHY: every call the model makes in thinking mode produces a reasoning trace,
# and until now we threw all of it away and diagnosed failures from a one-letter
# pick plus a margin. The trace is the only channel that says WHY the pick
# happened, so it is the raw material for the failure interview and for the
# cheap deterministic tells (probes/trace_autopsy.py).
#
# INVARIANTS (all three are pinned by tests):
#   1. capture NEVER affects the call result — every step is inside try/except.
#   2. knob off (kernel.capture_traces=False) writes NOTHING, not even the dir.
#   3. the certificate carries a REF (file + line index) and a token count,
#      never the trace body — traces are megabytes, certs are read by hand.
TRACES_DIR = pathlib.Path(__file__).resolve().parent / "results" / "traces"

_TRACE = threading.local()          # per-thread: last written ref (workers > 1)
_TRACE_CAPTURE = None               # None = not configured -> gym_config default
_TRACE_KEY = None                   # correlation key; None -> run-at key
_TRACE_LOCK = threading.Lock()      # one appender at a time per process
_TRACE_COUNTS = {}                  # trace file -> next line index

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)


def configure_traces(cfg=None, *, key=None, enabled=None):
    """Set the capture knob and the correlation key for this process.

    Purely additive: never called => knob falls back to the gym_config default
    and the key falls back to a run-at stamp.
    """
    global _TRACE_CAPTURE, _TRACE_KEY
    if cfg is not None:
        try:
            _TRACE_CAPTURE = bool(cfg.kernel.capture_traces)
        except AttributeError:
            pass
    if enabled is not None:
        _TRACE_CAPTURE = bool(enabled)
    if key is not None:
        _TRACE_KEY = str(key)
    return _TRACE_CAPTURE


def _trace_enabled():
    if _TRACE_CAPTURE is not None:
        return _TRACE_CAPTURE
    try:
        return bool(gym_config.default().kernel.capture_traces)
    except Exception:
        return False


def trace_key():
    """The correlation key traces are filed under (stable for the process)."""
    global _TRACE_KEY
    if not _TRACE_KEY:
        _TRACE_KEY = "runat-%s-%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    return _TRACE_KEY


def last_trace_ref():
    """{'file':..., 'line':..., 'tokens':...} for THIS thread's last captured
    call, or None. Read by the menu layer to stamp the certificate."""
    return getattr(_TRACE, "ref", None)


def _extract_trace(resp):
    """(trace_text, shape) from a completion response.

    llama.cpp thinking mode ships the reasoning in one of two shapes depending
    on build/flags, so BOTH are handled and the shape is recorded:
      * "reasoning_content" — a sibling field of message.content
      * "inline_think"      — a <think>...</think> block inside message.content
    Returns ("", "") when there is no trace (a non-thinking response).
    """
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return "", ""
    if not isinstance(msg, dict):
        return "", ""
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc, "reasoning_content"
    content = msg.get("content")
    if isinstance(content, str):
        blocks = _THINK_RE.findall(content)
        if blocks:
            joined = "\n".join(b.strip() for b in blocks if b.strip())
            if joined:
                return joined, "inline_think"
        # UNTERMINATED: the model ran out of budget mid-thought and never closed
        # the tag. This is the starvation case the digest exists to find, so it
        # must be captured, not dropped — and it is a DISTINCT shape.
        tail = _open_think(content)
        if tail:
            return tail, "inline_think_unterminated"
    return "", ""


def _open_think(text):
    """Text after an unclosed ``<think>``, or '' when the tag is closed/absent."""
    try:
        low = (text or "").lower()
        i = low.rfind("<think>")
        if i < 0 or "</think>" in low[i:]:
            return ""
        return text[i + len("<think>"):].strip()
    except Exception:
        return ""


def _strip_think(text):
    """Answer text with any inline <think> block removed (never raises).

    An UNCLOSED <think> means the reply is pure reasoning with no answer in it;
    everything from the open tag on is dropped, which usually leaves "" and so
    correctly trips _call's bigger-budget retry instead of parsing a thought.
    """
    try:
        out = _THINK_RE.sub("", text or "")
        low = out.lower()
        i = low.rfind("<think>")
        if i >= 0 and "</think>" not in low[i:]:
            out = out[:i]
        return out.strip()
    except Exception:
        return text or ""


def _capture_trace(resp, label=None):
    """Append one trace line; record the ref for this thread. FAIL-SOFT.

    Any failure here (bad response shape, unwritable disk, encoding) leaves the
    call result untouched and simply clears this thread's ref.
    """
    _TRACE.ref = None
    try:
        if not _trace_enabled():
            return
        trace, shape = _extract_trace(resp)
        if not trace:
            return
        try:
            answer = (resp["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            answer = ""
        # cheap token estimate: whitespace words. The trace budget question is
        # "did it run out of room", which words answer well enough.
        tokens = len(trace.split())
        rec = {"label": label, "ts": time.time(), "shape": shape,
               "budget_used_estimate": tokens, "trace": trace,
               "answer_raw": _strip_think(answer)[:200]}
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACES_DIR / (trace_key() + ".jsonl")
        line = json.dumps(rec, ensure_ascii=False)
        key = str(path)
        with _TRACE_LOCK:
            # line index = 0-based position of the record just written. Counted
            # ONCE per file then carried in memory: re-reading a multi-megabyte
            # trace file on every call would make capture quadratic.
            if key not in _TRACE_COUNTS:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        _TRACE_COUNTS[key] = sum(1 for _ in fh)
                except OSError:
                    _TRACE_COUNTS[key] = 0
            with open(path, "a", encoding="utf-8") as fh:
                # BYTE OFFSET is the only exact, O(1), multi-writer-safe address
                # for the record about to be written: seek-to-end reads the real
                # file state, where the in-memory line counter only knows about
                # THIS process. A reader that seeks to `offset` and reads one
                # line gets this record no matter who else appended.
                fh.seek(0, os.SEEK_END)
                offset = fh.tell()
                fh.write(line + "\n")
            idx = _TRACE_COUNTS[key]
            _TRACE_COUNTS[key] = idx + 1
        # `line` is a ZERO-BASED index into the jsonl, and `index_base` says so
        # in the record itself. MEASURED 2026-08-29 against the armed cycle's
        # ledger (results/traces/runat-20260828T225341-85867.jsonl): 9 of 9
        # verifiable refs dereference to the record whose `answer_raw` equals the
        # margin's `raw_response` — 0 off-by-one. An autopsy reported the exact
        # opposite ("0/63 aligned, every ref one low"); it was comparing this
        # 0-based index against 1-based file line numbers. The accounting is not
        # changed on that report — it is pinned by
        # test_trace_ref_round_trips_to_its_own_record and made self-describing
        # here so the same misreading cannot happen twice.
        _TRACE.ref = {"file": str(path), "line": idx, "index_base": 0,
                      "offset": offset, "tokens": tokens, "shape": shape}
    except Exception:
        _TRACE.ref = None


# ------------------------------------------------- structural answer channel
# STRUCTURAL ANSWER CHANNEL (ladder review 2026-08-29, ranked fix #2).
# 27 of 84 ladder runs — a third of everything — recorded
# pick=None not because the model had no answer but because it wrote the answer
# as PROSE, or spilled its reasoning into `content`. Several of those traces hold
# the CORRECT answer in plain English ("The correct fix is to change the
# condition so that `passed` is set to true when the score is above the thr").
# Parsing harder is how we got the lenient ladder; the fix is to make the wrong
# shape UNREPRESENTABLE. llama.cpp accepts a GBNF `grammar` in the request body,
# so a menu call can be constrained to emit exactly {"pick": "<letter>"}.
#
# FAIL-OPEN BY CONSTRUCTION: a server that rejects the key (any transport error
# on a grammar-carrying body) is retried ONCE with the key removed, and the
# process then remembers that grammar is unsupported so no later call pays the
# same double cost. A server that silently ignores the key behaves exactly as it
# does today.
_GRAMMAR_UNSUPPORTED = False        # set once, process-wide, on first rejection

# ------------------------------------------------- REASONING HEADROOM (#11d)
# Every call site here used to size `max_tokens` as if it bought OUTPUT. It does
# not: llama.cpp counts the model's THINKING inside the same budget, and this 1B
# thinks ~260 words (~350 tokens) before it writes anything — measured across all
# four ladder arms, flat, from `budget_used_estimate` in the trace files.
#
# So a site asking for 64 "answer" tokens was really giving the model 64 tokens
# TOTAL, and the model spent every one of them reasoning. The receipts are in the
# C5 traces: answers that stop mid-word — "Among the candida", "so `passed` is
# set to true when the score is above the thr" — with the CORRECT answer visibly
# forming. Those were recorded as `pick=None` -> ABSTAIN, which is a lie: the
# model did not decline, it was guillotined. 27 of 84 ladder runs died this way.
#
# Two halves, and BOTH are required (production rule #11d):
#   (a) every budget is reasoning_headroom + answer_floor, never answer-only;
#   (b) `finish_reason == "length"` is a FAILURE, never a parseable answer.
# Grammar-constrained calls make the ANSWER short; headroom makes the THINKING
# that precedes it survivable. They stack; neither replaces the other.
#   (c) the headroom must SCALE WITH THE SERVER'S REASONING BUDGET or a ladder
#       run re-strangles thinking at the client (E63's exact bug, one layer in:
#       a fixed 600 caps every rung above it). REASONING_BUDGET env is the same
#       variable the gym stamps into row config, so client and server agree.
# The forensic p90 wall is ~1,839 characters of thought and recent stages spend
# materially more than the old 600-token floor before emitting an answer. 1,536
# tokens covers that measured tail while the bounded ladder handles outliers.
REASONING_HEADROOM = 1536         # measured p90-sized hidden thought floor
REASONING_MULTIPLIER = 2.0        # retain margin when server budget is raised

# ------------------------------------------------ SLOT CLAMP (stacked autopsy)
# MEASURED on `3de6826bbea7`: 32 of 84 votes came back `finish_reason: "length"`
# and 27 of 78 carried `raw_response: ""`. The headroom fix above bought the
# model room to THINK and, on the wide file menu, it spent every token of it —
# 2,443 / 2,704 / 2,951-token traces that loop a sentence verbatim and never
# emit. Headroom without a ceiling is not thinking room, it is rope.
#
# So the client budget is additionally clamped to the SLOT the call actually
# occupies: whatever the arithmetic above asks for, an ANSWER must still be
# affordable inside `SLOT_CTX`. Estimate is deliberately crude and SAFE
# (`len(prompt) // 3` over-counts tokens for English/code), and the reserve is
# the same 64 tokens the continuation re-ask uses.
SLOT_ANSWER_RESERVE = 64            # tokens kept back for the visible answer
SLOT_FLOOR = 128                    # never clamp a call into uselessness


def _slot_ctx():
    return int(os.environ.get("SLOT_CTX", "4096") or 4096)


def prompt_token_estimate(prompt):
    """Crude, deliberately HIGH token estimate for a prompt (3 chars/token)."""
    return len(prompt or "") // 3


def clamp_to_slot(max_tokens, prompt):
    """(budget, clamped) — `max_tokens` capped so an answer fits in SLOT_CTX."""
    room = _slot_ctx() - prompt_token_estimate(prompt) - SLOT_ANSWER_RESERVE
    room = max(SLOT_FLOOR, room)
    if max_tokens <= room:
        return int(max_tokens), False
    return int(room), True


def budget_for(answer_tokens, prompt=None, *, reasoning_headroom=None,
               reasoning_multiplier=None, enable_thinking=True):
    """`max_tokens` for a site that needs `answer_tokens` of visible answer.

    Thinking calls receive a measured floor plus a server-budget-scaled margin;
    no-thinking calls receive only their visible-answer budget. Both are clamped
    to the prompt slot so the answer remains affordable.
    """
    try:
        server_budget = int(os.environ.get("REASONING_BUDGET", "400") or 400)
    except (TypeError, ValueError):
        server_budget = 400
    if enable_thinking:
        floor = REASONING_HEADROOM if reasoning_headroom is None else max(
            0, int(reasoning_headroom))
        multiplier = (REASONING_MULTIPLIER if reasoning_multiplier is None
                      else max(0.0, float(reasoning_multiplier)))
        headroom = max(floor, int(server_budget * multiplier))
    else:
        headroom = 0
    total = headroom + max(0, int(answer_tokens))
    if prompt is None:
        return total
    return clamp_to_slot(total, prompt)[0]


_STAGE_DEFAULTS = {
    "menu": (96, True),
    "relevance": (16, True),
    "binary_probe": (16, True),
    "line_probe": (16, True),
    "line_ballot": (16, True),
    "staged_reasoning": (128, True),
    "line_emission": (400, True),
    "hole": (32, True),
    "block": (320, True),
    "microcall": (32, False),
    "reask": (8, False),
}


def _stage_settings(cfg, stage):
    """Return `(answer_tokens, thinking)` with legacy-config fallbacks."""
    key = str(stage or "menu").strip().lower()
    default_answer, default_thinking = _STAGE_DEFAULTS.get(
        key, _STAGE_DEFAULTS["menu"])
    llm = getattr(cfg, "llm", None)
    answer = getattr(llm, "%s_answer_tokens" % key, default_answer)
    thinking_name = "%s_thinking" % key
    thinking = getattr(llm, thinking_name, default_thinking)
    if key in ("hole", "block"):
        thinking = getattr(llm, "authoring_thinking", thinking)
    return max(0, int(answer)), bool(thinking)


def budget_for_stage(cfg, stage, prompt=None, answer_tokens=None):
    """Config-routed budget for one named model-call stage."""
    answer, thinking = _stage_settings(cfg, stage)
    if answer_tokens is not None:
        answer = max(0, int(answer_tokens))
    llm = getattr(cfg, "llm", None)
    headroom = getattr(llm, "reasoning_headroom", REASONING_HEADROOM)
    multiplier = getattr(llm, "reasoning_multiplier", REASONING_MULTIPLIER)
    return budget_for(answer, prompt, reasoning_headroom=headroom,
                      reasoning_multiplier=multiplier,
                      enable_thinking=thinking)


def stage_thinking(cfg, stage):
    """Whether a named stage is allowed to use model reasoning."""
    return _stage_settings(cfg, stage)[1]


# The continuation instruction. Deliberately tells the model its thinking is
# SPENT: the measured failure is not an absent opinion, it is an opinion that
# never got written down (traces name the gold file, then loop until the cap).
CONTINUATION_Q = ('You have used your thinking. Reply now with only the menu '
                  'answer object: {"pick": "<letter>"}.')
CONTINUATION_RELEVANCE_Q = (
    'You have used your thinking. Reply now with only: '
    '{"answer": "<yes|no|unsure>"}.')
CONTINUATION_BINARY_Q = (
    'You have used your thinking. Reply now with only: '
    '{"answer": "<yes|no>"}.')
CONTINUATION_LINE_MENU_Q = (
    'You have used your thinking. Reply now with only: '
    '{"status": "selected", "line_ids": ["<offered_id>"]}.')
CONTINUATION_LINE_PROBE_Q = (
    'You have used your thinking. Reply now with only: '
    '{"answer": "<yes|no|uncertain>"}.')
CONTINUATION_LINE_BALLOT_Q = (
    'You have used your thinking. Reply now with only: '
    '{"bug_here": "<yes|no>"}.')
CONTINUATION_LINE_EMISSION_Q = (
    'You have used your thinking. Reply now with exactly one replacement source '
    'line, with no explanation, line number, or code fence.')
CONTINUATION_STAGED_Q = (
    'You have used your thinking. Reply now with exactly two complete reasoning '
    'sentences, and do not output JSON or a repair.')
CONTINUATION_BLOCK_Q = (
    'You have used your thinking. Reply now with only one complete block of '
    'source code in the requested indentation, with no explanation.')
CONTINUATION_FILL_Q = (
    'You have used your thinking. Reply now with only one fill value matching '
    'the requested hole type, with no explanation.')
CONTINUATION_MICROCALL_Q = (
    'You have used your thinking. Reply now with only: '
    '{"answer": "<offered_id>"}.')
CONTINUATION_REASK_Q = (
    'You have used your thinking. Reply now with only: '
    '{"pick": "<letter>"}.')

# Per-call telemetry the menu layer reads back. Thread-local because the gym
# runs workers > 1 and a module global would cross-attribute clamps.
_CALL_META = threading.local()


def _call_meta_reset():
    _CALL_META.slot_clamped = False
    _CALL_META.continuation_used = False
    _CALL_META.usage = None


def call_meta():
    """{'slot_clamped':bool, 'continuation_used':bool, 'usage':dict|None} for
    THIS thread's last `_call`. Empty-ish defaults when no call has run on
    this thread. `usage` is {'completion_tokens':int,'prompt_tokens':int} for
    THIS SPECIFIC call (not a running total) -- None when the server's
    response carried no usage block."""
    return {"slot_clamped": bool(getattr(_CALL_META, "slot_clamped", False)),
            "continuation_used": bool(
                getattr(_CALL_META, "continuation_used", False)),
            "usage": getattr(_CALL_META, "usage", None)}


# ------------------------------------------------------ TOKEN THROUGHPUT
# One run's total completion/prompt tokens + call count + measured wall time,
# so the dashboard can chart tokens/sec. FAIL-SOFT throughout: a response with
# no 'usage' block (or a malformed one) simply contributes nothing -- it never
# raises and never poisons the running total.
#
# The sink is a MUTABLE DICT the caller owns (same pattern as `cert["margins"]
# = fx.margins` above it): `set_usage_sink(d)` points this thread's accumulator
# at `d`, and every `_call` on this thread mutates it in place. That means the
# certificate can carry `cert["usage_totals"] = usage_totals_dict` BY
# REFERENCE from the top of `run()` and every one of its many early returns
# already carries the right numbers -- no return-point-by-return-point wiring.
_USAGE_SINK = threading.local()
_TRUNCATION_SINK = threading.local()


def _new_truncation_stats():
    """Mutable per-run model-attempt telemetry for the certificate."""
    return {"attempts": 0, "truncated": 0, "retries": 0,
            "finish_reason_counts": {}, "truncation_rate": 0.0}


def set_truncation_sink(stats):
    """Point this thread's truncation telemetry at a caller-owned dict."""
    _TRUNCATION_SINK.stats = stats


def truncation_stats():
    """Return the current thread's truncation telemetry, or empty defaults."""
    stats = getattr(_TRUNCATION_SINK, "stats", None)
    return stats if isinstance(stats, dict) else _new_truncation_stats()


def _record_call_outcome(resp, *, retry=False):
    """Record one completed model response without affecting its result."""
    stats = getattr(_TRUNCATION_SINK, "stats", None)
    if not isinstance(stats, dict):
        return
    try:
        reason = _finish_reason(resp) or "unknown"
        counts = stats.setdefault("finish_reason_counts", {})
        counts[reason] = int(counts.get(reason, 0)) + 1
        stats["attempts"] = int(stats.get("attempts", 0)) + 1
        if reason == "length":
            stats["truncated"] = int(stats.get("truncated", 0)) + 1
        if retry:
            stats["retries"] = int(stats.get("retries", 0)) + 1
        attempts = stats["attempts"]
        stats["truncation_rate"] = (
            float(stats["truncated"]) / attempts if attempts else 0.0)
    except Exception:
        # Telemetry cannot make a model response fail.
        return


def _new_usage_totals():
    return {"completion_tokens": 0, "prompt_tokens": 0, "calls": 0,
            "gen_seconds_estimate": 0.0}


def set_usage_sink(totals):
    """Point THIS thread's usage accumulation at `totals` (a dict this thread
    now owns and mutates in place, typically `cert["usage_totals"]`)."""
    _USAGE_SINK.totals = totals


def _record_usage(resp, wall_seconds):
    """FAIL-SOFT: extract usage from `resp`, stash it as THIS call's meta
    (read back by `call_meta()['usage']`), and accumulate into the current
    thread's usage sink, if one is set. Never raises."""
    try:
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if not isinstance(usage, dict):
            return
        ct = usage.get("completion_tokens")
        pt = usage.get("prompt_tokens")
        ct = int(ct) if isinstance(ct, (int, float)) and not isinstance(ct, bool) else None
        pt = int(pt) if isinstance(pt, (int, float)) and not isinstance(pt, bool) else None
        if ct is None and pt is None:
            return
        _CALL_META.usage = {"completion_tokens": ct or 0, "prompt_tokens": pt or 0}
        totals = getattr(_USAGE_SINK, "totals", None)
        if not isinstance(totals, dict):
            return
        totals["completion_tokens"] = totals.get("completion_tokens", 0) + (ct or 0)
        totals["prompt_tokens"] = totals.get("prompt_tokens", 0) + (pt or 0)
        totals["calls"] = totals.get("calls", 0) + 1
        totals["gen_seconds_estimate"] = (
            totals.get("gen_seconds_estimate", 0.0) + max(0.0, float(wall_seconds or 0.0)))
    except Exception:
        pass


def _timed_post(body, url, *, label=None):
    """Call `_post`, measure it, and capture the complete attempt centrally.

    This is deliberately below every `_call` retry/fallback branch: a failed
    grammar attempt, normal retry, and continuation are all model calls and get
    independent ledger rows.  Ledger capture is observability and cannot alter
    transport behavior.
    """
    t0 = time.time()
    try:
        resp = _post(body, url)
    except Exception as exc:
        wall = time.time() - t0
        try:
            prompt_ledger.capture_call(body=body, response=None,
                                       wall_time_seconds=wall, label=label,
                                       error=exc)
        except Exception:
            pass
        raise
    wall = time.time() - t0
    try:
        prompt_ledger.capture_call(body=body, response=resp,
                                   wall_time_seconds=wall, label=label)
    except Exception:
        pass
    return resp, wall


def _answer_text(resp):
    """`message.content` of a completion, or '' — never raises."""
    try:
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _finish_reason(resp):
    """The completion's finish_reason, or '' when unavailable."""
    try:
        return str(resp["choices"][0].get("finish_reason") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def pick_grammar(letters):
    """GBNF whose ONLY legal completion is `{"pick": "<one of letters>"}`."""
    ls = [str(l)[:1].upper() for l in letters if str(l)[:1].strip()]
    if not ls:
        return None
    alts = " | ".join('"%s"' % l for l in ls)
    return 'root ::= "{\\"pick\\": \\"" letter "\\"}"\nletter ::= %s\n' % alts


def _call(prompt, *, seed, url, max_tokens=budget_for(96), n_probs=0,
          label=None, grammar=None, enable_thinking=None, continuation=None):
    """client2.ask semantics (content-only, fence-strip, retry-on-empty/degenerate)
    but returns the RAW response too so menu picks can read logprobs.

    `grammar` (GBNF text) constrains the completion shape. It is dropped
    automatically, once and for the life of the process, against any server that
    will not accept it — the call is never lost to an unsupported knob.
    """
    global _GRAMMAR_UNSUPPORTED
    last = None
    _call_meta_reset()
    rungs = []
    for raw_budget in (max_tokens, max_tokens * 2, max_tokens * 3):
        b, clamped = clamp_to_slot(raw_budget, prompt)
        if clamped:
            _CALL_META.slot_clamped = True
        if b not in rungs:                  # a clamped ladder repeats itself
            rungs.append(b)
    for budget in rungs:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "top_p": 1, "max_tokens": budget, "seed": seed}
        if n_probs:
            # OpenAI-compat chat endpoint wants logprobs/top_logprobs; n_probs is
            # the NATIVE /completion param and is silently ignored here (the
            # margin:null bug in the first full-stack smoke).
            body["logprobs"] = True
            body["top_logprobs"] = n_probs
        if enable_thinking is not None:
            # MiniCPM's prompt-level "answer only yes/no" instruction does not
            # disable its reasoning template. The template flag is the actual
            # no-think control, and is intentionally opt-in so ordinary fixer
            # calls retain their existing reasoning behavior.
            body["chat_template_kwargs"] = {
                "enable_thinking": bool(enable_thinking)}
        use_grammar = bool(grammar) and not _GRAMMAR_UNSUPPORTED
        if use_grammar:
            body["grammar"] = grammar
        try:
            resp, _wall = _timed_post(body, url, label=label)
        except Exception:
            if not use_grammar:
                raise
            # The grammar key is the only thing this body carries that a plain
            # OpenAI-compatible server may refuse. Retry without it, once, and
            # remember — a permanently unsupported knob must not double every
            # menu call for the rest of the run.
            _GRAMMAR_UNSUPPORTED = True
            body.pop("grammar", None)
            resp, _wall = _timed_post(body, url, label=label)
        _retryable_length = (
            _finish_reason(resp) == "length" and budget != rungs[-1])
        _record_call_outcome(resp, retry=_retryable_length)
        _record_usage(resp, _wall)
        last = resp
        _capture_trace(resp, label=label)
        try:
            c = (resp["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            c = ""
        # An inline <think> block is REASONING, not the answer. Leaving it in
        # lets a stray '"pick": "B"' inside the model's deliberation win the
        # downstream regex, and makes a think-only (budget-starved) reply look
        # like a real answer instead of triggering the bigger-budget retry.
        c = _strip_think(c)
        # #11d(b): a TRUNCATED completion is a failure, not an answer. The
        # budget ladder above already doubles and triples, but it only ever
        # retried on empty/degenerate output — so a reply guillotined mid-word
        # ("Among the candida") was parsed as if the model had finished, and its
        # missing letter was recorded as an abstention. Every site now retries on
        # `length`, whatever max_tokens it passed.
        if _retryable_length:
            continue
        if _finish_reason(resp) == "length":
            continue
        if not c or degenerate(c):
            continue
        lines = [l for l in c.splitlines() if not l.strip().startswith("```")]
        return "\n".join(lines).strip(), resp
    # ---- FORCED EMISSION (stacked-autopsy half 1a) -------------------------
    # The ladder is exhausted and the model is STILL mid-thought. That is not a
    # refusal and it is not a datum: `finish_reason == "length"` with no content
    # is a FAILED CALL (#11d). Ask ONE more time, on the same question, with the
    # assistant's partial deliberation DROPPED and the thinking declared over.
    # 64 tokens is not enough to think in, which is the entire point.
    # "empty OR unparseable": an unterminated `<think>` block is content the
    # answer parser can never read, so it counts as empty here too.
    if (_finish_reason(last) == "length"
            and not _strip_think(_answer_text(last))
            and isinstance(continuation, str) and continuation.strip()):
        try:
            body = {"messages": [{"role": "user", "content": prompt},
                                 {"role": "user", "content": continuation}],
                    "temperature": 0, "top_p": 1,
                    "max_tokens": SLOT_ANSWER_RESERVE, "seed": seed,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if grammar and not _GRAMMAR_UNSUPPORTED:
                body["grammar"] = grammar
            if n_probs:
                body["logprobs"] = True
                body["top_logprobs"] = n_probs
            resp, _wall = _timed_post(
                body, url, label=(label or "") + ":continuation")
            _record_call_outcome(resp)
            _record_usage(resp, _wall)
            _CALL_META.continuation_used = True
            _capture_trace(resp, label=(label or "") + ":continuation")
            c = _strip_think(_answer_text(resp))
            if c and not degenerate(c):
                lines = [l for l in c.splitlines()
                         if not l.strip().startswith("```")]
                return "\n".join(lines).strip(), resp
            return "", resp
        except Exception:
            _CALL_META.continuation_used = True      # attempted and lost
    # A final `length` response is never parseable, even when it contains a
    # plausible prose fragment; callers must see the finish reason and abstain.
    if _finish_reason(last) == "length":
        return "", last
    return "", last


def ask(prompt, **kw):
    """Back-compat wrapper: Reply-shaped, routed through the same seam."""
    kw.setdefault("url", URL)
    kw.setdefault("seed", 1)
    text, _ = _call(prompt, **kw)
    return Reply(text, bool(text), "" if text else "empty")


# Punctuation that realistically travels ATTACHED to the answer letter in a
# llama.cpp tokenization of `{"pick": "A"}` / `A)` / `**A**`. Stripping these
# from both ends is what turns a multi-char token back into a bare letter.
# Anything left with more than one character is NOT the answer token.
_TOK_TRIM = ' \t\r\n"\'`*{}[]()<>:,.;!?-_'


def _norm_letter_token(raw):
    """The bare letter a response token carries, or '' if it carries none."""
    return str(raw or "").strip(_TOK_TRIM).upper()


def _token_strings(resp, limit=3):
    """First ``limit`` raw token strings, for margin_debug. [] if unavailable."""
    try:
        toks = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError, IndexError):
        return []
    if not isinstance(toks, list):
        return []
    return [str(t.get("token"))[:24] for t in toks[:limit] if isinstance(t, dict)]


def _letter_margin(resp, letters):
    """Top-2 logprob gap (nats) on the answer LETTER token. None = unavailable.

    Same shape as detreason/kernel.py:LlamaSensor._margin, retargeted from the
    yes/no/unknown token to the menu letter. The letter rarely arrives as a bare
    token — it comes wrapped as '"A', 'A"}', ' A', 'A)' depending on how the
    model spelled the JSON — so candidates are trimmed of attached punctuation
    before matching. A letter buried inside a real word ('Answer', '{"pick"')
    still does not match: after trimming it must be exactly one letter.
    """
    try:
        toks = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError, IndexError):
        return None
    if not isinstance(toks, list):
        return None
    want = {l.upper() for l in letters}
    for tok in toks:
        if not isinstance(tok, dict):
            continue
        t = _norm_letter_token(tok.get("token"))
        if t in want:
            tops = sorted((x["logprob"] for x in tok.get("top_logprobs", [])),
                          reverse=True)
            if len(tops) >= 2:
                return tops[0] - tops[1]        # nats
            return None
    return None


def _lenient_letter(text, letters):
    """LENIENT MENU PARSE (at least five gym tasks died on 'menu abstained'
    while the margin was >1.0 — the model answered `A`, `A)`, `**A**` or an
    unquoted `{"pick": A}` and the strict `"pick": "A"` regex saw nothing).

    Normalization ladder, applied ONLY after the strict parse has failed:
      1. unquoted/loosely-quoted JSON letter value
      2. strip markdown emphasis + punctuation + whitespace; accept iff exactly
         ONE character remains and it is a valid letter
    NEVER guesses between two letters — anything that does not resolve to exactly
    one in-range letter returns None and the menu still abstains."""
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r'"?\bpick"?\s*:\s*"?([A-Za-z])\b"?', t)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    cleaned = re.sub(r"[*_`#~\s]", "", t).strip("{}[]()<>\"'.,:;!-")
    if len(cleaned) == 1 and cleaned.upper() in letters:
        return cleaned.upper()
    return None


def _prose_letter(text, letters):
    """Extract one explicit menu decision from a natural-language answer.

    A rationale may mention several candidate letters, so only explicit
    decision phrases or a line-labelled answer are eligible; ambiguous prose
    remains an abstention.
    """
    source = str(text or "")
    if not source.strip():
        return None
    allowed = {str(letter).upper() for letter in letters}

    explicit_patterns = (
        r"\b(?:final\s+)?(?:answer|choice|pick|selection|decision|verdict|letter)"
        r"\s*(?:is|:|=|should\s+be|must\s+be)?\s*"
        r"(?:option\s+)?([A-Z])\b",
        r"\b(?:choose|select|recommend(?:ed)?)\s+(?:option\s+)?([A-Z])\b",
        r"\b(?:correct|best)\s+(?:answer|choice|option|pick)\s*"
        r"(?:is|:|=)?\s*(?:option\s+)?([A-Z])\b",
    )
    for pattern in explicit_patterns:
        found = {m.group(1).upper() for m in re.finditer(pattern, source,
                                                           flags=re.I)}
        found &= allowed
        if len(found) == 1:
            return next(iter(found))
        if len(found) > 1:
            return None

    labelled = []
    for line in source.splitlines():
        match = re.match(r"^\s*(?:final\s+)?(?:option\s+)?([A-Z])\s*"
                         r"[)\].:,-](?:\s|$)", line, flags=re.I)
        if match and match.group(1).upper() in allowed:
            labelled.append(match.group(1).upper())
    labels = set(labelled)
    return next(iter(labels)) if len(labels) == 1 else None


def _prose_yes_no(text):
    """Extract one explicit yes/no verdict from a natural-language answer.

    Binary probes often return a short rationale even when the prompt requests
    one token.  Only a leading verdict or an explicit decision phrase counts;
    an answer that mentions both classes remains unreadable.
    """
    source = str(text or "")
    if not source.strip() or re.search(r"\b(?:yes|no)\s+or\s+(?:yes|no)\b",
                                      source, flags=re.I):
        return None
    patterns = (
        r"[\"']?(?:answer|decision|verdict|choice)[\"']?\s*:\s*"
        r"[\"']?(yes|no)\b",
        r"\b(?:final\s+)?(?:answer|decision|verdict|choice)\s*"
        r"(?:is|:|=|should\s+be|must\s+be)\s*(yes|no)\b",
        r"^\s*(yes|no)\b",
        r"\b(?:therefore|thus|so|overall|ultimately)\s*[:,\-]?\s*(yes|no)\b",
    )
    for pattern in patterns:
        found = {match.group(1).lower()
                 for match in re.finditer(pattern, source, flags=re.I)}
        if len(found) == 1:
            return next(iter(found))
        if len(found) > 1:
            return None
    return None


def _yes_no_score(resp):
    """Signed confidence on a yes/no answer token: +p(yes) when the model answered
    yes, -p(no) when it answered no. None when no yes/no token was emitted.

    Comparable ACROSS independent calls (unlike a menu letter's top-2 margin,
    which is only meaningful within one menu), which is the whole point: it lets
    separate single-line questions be ranked against each other without ever
    putting the candidates in a positional list. The 1B carries a heavy yes-bias,
    so the absolute value is not a probability of anything — only the DIFFERENCE
    between two lines asked the identical question is used, and only when that
    difference clears the configured decisiveness threshold."""
    try:
        toks = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError, IndexError):
        toks = None
    if not isinstance(toks, list):
        verdict = _prose_yes_no(_answer_text(resp))
        return 1.0 if verdict == "yes" else -1.0 if verdict == "no" else None
    for tok in toks:
        if not isinstance(tok, dict):
            continue
        k = _norm_letter_token(tok.get("token")).lower()
        if k in ("yes", "no"):
            try:
                p = math.exp(float(tok.get("logprob")))
            except (TypeError, ValueError):
                return None
            return p if k == "yes" else -p
    verdict = _prose_yes_no(_answer_text(resp))
    return 1.0 if verdict == "yes" else -1.0 if verdict == "no" else None


def _yes_no_class(raw):
    """Return the normalized binary class for a tokenizer spelling, if any."""
    value = _norm_letter_token(raw).lower()
    return value if value in ("yes", "no") else None


def _logprob_probability(value):
    """Convert one finite logprob to probability; malformed values are absent."""
    try:
        logprob = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(logprob) or logprob == math.inf:
        return None
    try:
        return math.exp(logprob)
    except (OverflowError, ValueError):
        return None


def _yes_no_logit_margin_raw(resp):
    """Return ``(log P(YES) - log P(NO), status)`` at the answer position.

    ``top_logprobs`` describes alternatives at each generated-token position.
    The selected token is included because llama.cpp versions differ on whether
    it is repeated in that list. Distinct spellings such as ``" YES"`` and
    ``"YES"`` are separate probability mass within the same class; an exact
    duplicate of the selected spelling is counted only once.
    """
    try:
        toks = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError, IndexError):
        return None, "missing_logprobs"
    if not isinstance(toks, list):
        return None, "missing_logprobs"

    saw_answer = False
    answer_top_logprobs = False
    for tok in toks:
        if not isinstance(tok, dict):
            continue
        selected_class = _yes_no_class(tok.get("token"))
        # The generated token identifies the answer position. Do not interpret
        # YES/NO alternatives attached to a preceding JSON/punctuation token as
        # an answer merely because they were among that position's candidates.
        if selected_class is None:
            continue
        saw_answer = True
        alternatives = tok.get("top_logprobs")
        if not isinstance(alternatives, list) or not alternatives:
            continue
        if selected_class is not None:
            answer_top_logprobs = True
        entries = []
        if "token" in tok:
            entries.append((tok.get("token"), tok.get("logprob")))
        entries.extend(
            (alt.get("token"), alt.get("logprob"))
            for alt in alternatives if isinstance(alt, dict))
        masses = {"yes": 0.0, "no": 0.0}
        seen_tokens = set()
        for token, logprob in entries:
            probability = _logprob_probability(logprob)
            if probability is None:
                continue
            token_key = repr(token)
            if token_key in seen_tokens:
                continue
            seen_tokens.add(token_key)
            klass = _yes_no_class(token)
            if klass is not None:
                masses[klass] += probability
        if masses["yes"] > 0.0 and masses["no"] > 0.0:
            return (math.log(masses["yes"]) - math.log(masses["no"]),
                    "answer_token_margin")

    if saw_answer:
        return (None, "missing_yes_no_counterpart"
                if answer_top_logprobs else "missing_top_logprobs")
    return None, "no_yes_no_token"


def _yes_no_scores(resp, scale=4.0):
    """Return ``(token_score, margin_score, raw_margin, status)``.

    ``margin_score`` is the monotone bounded form used by the selector, while
    ``raw_margin`` keeps the actual logit difference auditable in certificates.
    ``None`` margin values mean the requested logprob evidence was unavailable.
    """
    token_score = _yes_no_score(resp)
    raw_margin, status = _yes_no_logit_margin_raw(resp)
    if raw_margin is None:
        return token_score, None, None, status
    try:
        numeric_scale = float(scale)
        if not math.isfinite(numeric_scale) or numeric_scale == 0.0:
            raise ValueError("invalid margin scale")
        margin_score = math.tanh(raw_margin / numeric_scale)
    except (TypeError, ValueError, ZeroDivisionError):
        return token_score, None, None, "invalid_margin_scale"
    return token_score, margin_score, raw_margin, status


def _yes_no_margin(resp, scale=4.0):
    """Return the bounded YES-vs-NO margin, falling back to token scoring."""
    token_score, margin_score, _raw_margin, status = _yes_no_scores(resp, scale)
    if margin_score is not None:
        return margin_score, status
    # Keep the legacy helper's public provenance label stable; the selector uses
    # `_yes_no_scores` directly and records the precise fallback reason in its
    # certificate.
    return token_score, ("emitted_probability" if token_score is not None
                         else status)


def _exec_tokens(text):
    """Executable-token signature of a source fragment: every token that is not a
    comment, a string/docstring, or layout. Raises on untokenizable input so the
    caller can fail OPEN."""
    import io, tokenize
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    if hasattr(tokenize, "ENCODING"):
        skip.add(tokenize.ENCODING)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in skip:
            continue
        out.append((tok.type, tok.string))
    return out


def _is_inert_change(old, new):
    """True iff the delta between two source fragments changes NOTHING
    executable — the edit lives entirely inside a comment or a string/docstring.
    FAIL OPEN: any tokenize failure returns False (candidate kept)."""
    if old == new:
        return False                      # not our call — enumerator's problem
    try:
        import textwrap
        return _exec_tokens(textwrap.dedent(old)) == _exec_tokens(textwrap.dedent(new))
    except Exception:
        return False


_CALLABLE_BUILTINS = frozenset((
    "isinstance", "issubclass", "len", "getattr", "setattr", "hasattr",
    "int", "str", "float", "bool", "list", "dict", "set", "tuple", "type",
    "min", "max", "sum", "abs", "any", "all", "sorted", "enumerate", "range",
    "print", "repr", "format", "round", "zip", "map", "filter", "callable",
    "iter", "next", "open",
))


def uncallable_call_target(span, old, new):
    """"" when admissible, else WHY the edit made a NON-CALLABLE the callee.

    MEASURED 2026-08-29 on gym task 2b5d3f3a (`normalize_generated_content`),
    the closest near-miss on the baseline board. The selected mutation was

        if isinstance(inner, dict) and "sections" in inner:
     -> if inner(inner, dict) and "sections" in inner:

    -- the callee `isinstance` replaced by `inner`, a local the same line has
    just established is a dict. It compiles, so the compile gate passed; the
    witness replay "supported" it; spec_exec shortlisted it; the binary
    selector scored it 0.93; and the run reported FIX_PROPOSED with
    ready_to_act=true and n_blocking=0 while scoring fixed=0. A confident
    proposal that would raise `TypeError: 'dict' object is not callable` the
    first time the branch executes is a FALSE GREEN, and every gate in the
    ladder was blind to it because each one asks a question this defect
    answers correctly.

    The check is deliberately narrow, and fails OPEN on anything it cannot
    prove: it fires only when the ORIGINAL callee is a known callable builtin
    and the replacement callee is a bare local name that is never called
    anywhere else in the function. A name that IS called elsewhere is a real
    function in scope and is none of this guard's business.
    """
    try:
        o = ast.parse((old or "").strip() + ("\n    pass"
                                             if (old or "").strip().endswith(":")
                                             else ""))
        nnode = ast.parse((new or "").strip() + ("\n    pass"
                                                 if (new or "").strip().endswith(":")
                                                 else ""))
    except Exception:
        return ""                       # unparseable is not evidence
    def _callees(tree):
        return [c.func.id for c in ast.walk(tree)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    old_c, new_c = _callees(o), _callees(nnode)
    if len(old_c) != len(new_c):
        return ""
    import textwrap as _tw
    called_in_fn = set()
    try:
        fn_tree = ast.parse(_tw.dedent("\n".join(span or ())))
        called_in_fn = set(_callees(fn_tree))
    except Exception:
        pass                            # no function context: builtin check only
    for a, b in zip(old_c, new_c):
        if a == b:
            continue
        if a in _CALLABLE_BUILTINS and b not in _CALLABLE_BUILTINS \
                and b not in called_in_fn:
            return ("the edit calls `%s`, which is never called anywhere in "
                    "this function and is not a builtin; the original called "
                    "`%s`" % (b, a))
    return ""


def prose_regions(src):
    """{0-based line -> [(start_col, end_col), ...]} covering every STRING and
    COMMENT token in a file, with a full-line span for the interior lines of a
    multi-line string.

    Tokenizing the WHOLE file is the point. `_is_inert_change` tokenizes the
    candidate's one line in isolation, and a line lifted out of a triple-quoted
    docstring tokenizes perfectly happily as bare NAMEs — which is exactly how
    the real-19 board's #8 shipped `"...for a lesson and generate..."` ->
    `"...for a lesson or generate..."` as a FIX_PROPOSED at full confidence,
    past compile, polarity, regression-assert and re-emit. Only the file's own
    token stream knows that line is prose. FAILS OPEN: an untokenizable file
    yields {} and nothing is filtered.
    """
    import io, tokenize
    regions = {}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return {}
    for tok in toks:
        if tok.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        # PROSE, NOT DATA. A comment is always prose. A string is prose when it
        # reads like a sentence — it contains whitespace, or it is triple-quoted
        # (a docstring or a block of prompt text). An identifier-shaped literal
        # like "contract_slug" is DATA: its value is executable meaning, and
        # `literal_substitution` repairs exactly those. Filtering them as prose
        # would delete the family that fixes them, which is the whole point of
        # judging the string rather than the token type.
        if tok.type == tokenize.STRING:
            body = tok.string.lstrip("rRbBuUfF")
            triple = body[:3] in ('"""', "'''")
            inner = body[3:-3] if triple else body[1:-1]
            if not triple and not re.search(r"\s", inner):
                continue
        (r0, c0), (r1, c1) = tok.start, tok.end
        for row in range(r0, r1 + 1):
            lo = c0 if row == r0 else 0
            hi = c1 if row == r1 else float("inf")
            regions.setdefault(row - 1, []).append((lo, hi))
    return regions


def _changed_span(old, new):
    """(start, end) character span of the single contiguous region in which two
    lines differ, measured against `old`. None when they are identical."""
    if old == new:
        return None
    a = 0
    while a < len(old) and a < len(new) and old[a] == new[a]:
        a += 1
    b = 0
    while (b < len(old) - a and b < len(new) - a
           and old[len(old) - 1 - b] == new[len(new) - 1 - b]):
        b += 1
    return a, len(old) - b


def _is_prose_edit(regions, li, old, new):
    """True iff the whole changed span of a candidate lies inside a string or a
    comment. Multi-line candidates (guard/clause insertion) are NEVER prose: they
    add executable statements, and their `new` is not a same-line rewrite."""
    if "\n" in new:
        return False
    spans = regions.get(li)
    if not spans:
        return False
    ch = _changed_span(old, new)
    if ch is None:
        return False
    s, e = ch
    if s >= e:                      # pure insertion: judge the insertion point
        e = s + 1
    return any(lo <= s and e <= hi for lo, hi in spans)


# The CLOSED fix rungs: template-assembled, compile-checked candidates. Only
# these may be rescued by the docstring-provenance channel — free-form emission
# never can. (autopsy fix #2)
_DOCSTRING_LINKED_RUNGS = frozenset(("mutation_menu", "witness_replay",
                                     "spec_exec"))

# Grammar, not evidence. See the evidence-linkage gate for why the widened
# `[a-z_0-9]{2,}` token class must not admit these.
_LINKAGE_STOP = frozenset((
    "if", "or", "and", "in", "is", "not", "def", "el", "se", "for", "no",
    "to", "of", "on", "as", "at", "by", "it", "be", "do", "so", "we",
    "none", "true", "false", "return", "elif", "else", "while", "pass",
))


# ===================================================== EVIDENCE CHANNEL: IDF
# `symptom_identifier_lexical_v1`, wired 2026-08-29.
#
# THE 5 UNVERIFIED. A correct fix whose changed line and whose ticket share no
# LITERAL token is refused by every existing linkage channel and self-downgrades
# to LOW_CONFIDENCE_PROPOSAL, which never applies. The existing channels are all
# EXACT-STRING: `cw & tw` needs the same spelling, `file_anchor` needs a named
# symbol, `function_docstring` needs a >=60-char quote, `executed_evidence`
# needs a synthesized discriminating test to have run.
#
# THIS CHANNEL IS ADDITIVE AND STRICTLY WEAKER. It never replaces a channel and
# never withdraws one: it is one more disjunct, it is recorded under its own
# name in `evidence_linkage_by`, and it is deliberately built so that the ONE
# case it must refuse — the real-19 attractor, a plausible edit on a hub file
# the ticket never described — it still refuses.
#
# THE CONSTRUCTION, and what each part is holding up:
#   * tokens come from the SYMPTOM and from the CHANGED IDENTIFIERS, split on
#     snake_case and camelCase, with NO stemming, NO synonyms and NO fuzzy
#     matching. Every one of those would buy recall by making the channel
#     unfalsifiable.
#   * weights are IDF over the CORPUS, so a token that appears in most files
#     carries almost nothing. This is the part that refuses the attractor: a
#     hub file's identifiers are common by definition.
#   * BOTH thresholds must hold — one rare-enough matched token, AND the match
#     must cover most of the symptom's weighted vocabulary. Either alone is a
#     coin flip; a common-token overlap or a one-word brush is not linkage.
#
# STRENGTH 1, AND THAT IS THE CEILING. Per the design: a lexical link clears
# "no deterministic link", it does NOT clear behavioural verification. It maps
# `unverified_claim -> linked, still behaviour-unverified`, and it may never on
# its own satisfy a promotion or behavioural gate.

#: `1000 * (1 + ln((N+1)/(df+1)))`, so ~1750 is a token in <=47% of files.
IDF_RARE_MIN = 1750
#: Matched weight must cover >= 60% of the symptom's total weight.
IDF_COVERAGE_NUM = 600
IDF_COVERAGE_DEN = 1000

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _changed_line_pair(changed):
    """``(old, new)`` for a change stage. FAIL-CLOSED to ``("", "")``.

    Only the RAW keys are read. The flattened `change` display string is
    deliberately NOT parsed as a fallback: " -> " is not a reserved sequence, so
    a line containing one would silently yield a wrong split — and a wrong split
    here manufactures "added identifiers" that were never added, which is
    evidence out of thin air. A stage with no raw pair simply gets no link.
    """
    if not isinstance(changed, dict):
        return "", ""
    old, new = changed.get("change_old"), changed.get("change_new")
    if not isinstance(old, str) or not isinstance(new, str):
        return "", ""
    return old, new


def split_identifier(tok):
    """`snake_case`/`camelCase` -> lowercase parts. No stemming, ever."""
    out = []
    for piece in str(tok).split("_"):
        for m in _CAMEL_RE.findall(piece):
            w = m.lower()
            if w and not w.isdigit():      # numeric-only tokens are ignored
                out.append(w)
    return out


def build_idf(files):
    """``{token: weight}`` over the corpus. Integer weights, so the thresholds
    are exact and a calibration cannot drift on float formatting."""
    import math
    n = max(1, len(files))
    df = {}
    for src in files.values():
        seen = set()
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src or ""):
            for w in split_identifier(tok):
                seen.add(w)
        for w in seen:
            df[w] = df.get(w, 0) + 1
    return {w: int(round(1000 * (1 + math.log((n + 1) / (c + 1)))))
            for w, c in df.items()}


def changed_identifier_tokens(old_line, new_line):
    """Tokens the edit ADDED, split. Removed and unchanged tokens are excluded.

    The design's rule, and the reason for it: an edit is evidence for the ticket
    only through what it PUT THERE. Counting the tokens it removed — or the ones
    it left alone — would let any edit inside a well-named function link to any
    ticket about that function, which is co-location wearing an evidence badge.
    """
    def toks(s):
        out = []
        for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s or ""):
            out.extend(split_identifier(t))
        return out
    old, new = toks(old_line), toks(new_line)
    from collections import Counter
    added = Counter(new) - Counter(old)
    return set(added)


def symptom_identifier_link(task_text, old_line, new_line, idf):
    """``(linked, receipt)`` for `symptom_identifier_lexical_v1`.

    ABSTAINS (returns False) on: no symptom tokens, no added identifiers, no
    exact overlap, only common-token overlap, or under 60% weighted coverage.
    Every one of those is a REFUSAL and none of them is an error: a channel that
    reported "no opinion" as failure would be pressure to loosen it.
    """
    receipt = {"channel": "symptom_identifier_lexical_v1", "strength": 1}
    if not idf:
        receipt["result"] = "abstain:no_idf"
        return False, receipt
    sym = set()
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task_text or ""):
        for w in split_identifier(t):
            if w not in FILE_RANK_BOILERPLATE and w not in _LINKAGE_STOP \
                    and len(w) > 2:
                sym.add(w)
    if not sym:
        receipt["result"] = "abstain:no_symptom_tokens"
        return False, receipt
    edited = changed_identifier_tokens(old_line, new_line)
    if not edited:
        receipt["result"] = "abstain:no_edited_identifier"
        return False, receipt
    matched = sym & edited
    total = sum(idf.get(w, 0) for w in sym)
    hit = sum(idf.get(w, 0) for w in matched)
    receipt.update({"matched": sorted(matched)[:8],
                    "max_weight": max((idf.get(w, 0) for w in matched),
                                      default=0),
                    "coverage_num": hit, "coverage_den": total})
    if not matched:
        receipt["result"] = "abstain:no_overlap"
        return False, receipt
    if receipt["max_weight"] < IDF_RARE_MIN:
        receipt["result"] = "abstain:only_common_tokens"
        return False, receipt
    if total <= 0 or hit * IDF_COVERAGE_DEN < IDF_COVERAGE_NUM * total:
        receipt["result"] = "abstain:coverage_below_60pct"
        return False, receipt
    receipt["result"] = "support"
    return True, receipt


def _task_quotes_docstring(task_text, node):
    """EVIDENCE LINKAGE channel: synthetic tasks are
    generated FROM the target function's docstring, so a correct fix whose changed
    line shares no token with the task text is still provably on-target. True iff
    the task text contains the function's docstring (whitespace-collapsed,
    casefolded) or a >=60-char contiguous chunk of it."""
    try:
        doc = ast.get_docstring(node)
    except Exception:
        return False
    if not doc:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
    d, t = norm(doc), norm(task_text)
    if not d or not t:
        return False
    if d in t:
        return True
    if len(d) < 60:
        return False
    return any(d[i:i + 60] in t for i in range(len(d) - 59))


# ---------------------------------------------------------------------------
# LINE-BALLOT VERDICT — pure, so it can be REPLAYED against recorded ballots.
#
# Extracted from run() deliberately: the ladder autopsy asked for a guard that
# measures ballot-selection's gold-line hit rate over the whole 21-task set
# BEFORE selection is allowed to ship, and a decision buried inside a 700-line
# function cannot be replayed against a ledger. Every knob is read here, nothing
# else decides, and `results/unified_gym.jsonl` already carries the recorded
# `scores`/`incumbent_line` this eats.
#
# Three outcomes, strictly ordered by how much power they hand the ballot:
#   selected*  the menu is FILTERED to the winning line's variants
#   reordered  the winning line's variants lead; every other line stays
#   kept: ...  nothing moves
#
# The gate between `reordered` and `selected` is the armed-cycle correction.
# Reordering is cheap to be wrong about — the losing line is still on the menu.
# Filtering is not: it hands the rung to the ballot outright. Measured on the
# armed cycle, the ballot is WRONG on 2 of the 4 chronic tasks:
#   image_processing  {307: 0.79, 312: 0.71}     gold 312, gap 0.08
#   curriculum        {456,458,460} all ~0.9996  gold 459, saturated on three
# so selection additionally requires no multi-line saturation (which carries no
# information at all) and a real margin at `select_tau`. Both counterexamples
# fall back to reorder-only under that gate, which is exactly the "disagreement
# escape hatch": when ballot and incumbent disagree without conviction, neither
# wins outright and both lines stay on the menu.
def ballot_verdict(scores, incumbent, cfg):
    """{decision, winner, gap, tau, select_tau, saturated_*} — pure."""
    tau = float(getattr(cfg.fix, "line_ballot_tau", 0.30))
    sel_tau = float(getattr(cfg.fix, "line_ballot_select_tau", 0.15))
    satv = float(getattr(cfg.fix, "line_ballot_saturation", 0.999))
    can_select = bool(getattr(cfg.fix, "line_ballot_select", False))
    sat_ok = bool(getattr(cfg.fix, "line_ballot_saturated_decisive", False))
    factor = float(getattr(cfg.fix, "line_ballot_select_factor", 2.0))
    out = {"tau": tau, "select_tau": sel_tau, "winner": None}
    if len(scores) < 2 or incumbent not in scores:
        out["decision"] = "kept: ballot unreadable"
        return out
    sat = [k for k, v in scores.items() if v >= satv]
    saturated_unique = len(sat) == 1 and sat_ok
    # FLAT FIELD = SATURATED FIELD (stacked-autopsy runner-up 3). The absolute
    # test above only sees saturation near +1; `log_handler`'s ballot scored
    # every probed line at -0.9998/-0.9997 — a field carrying no information
    # whatsoever — and sailed straight past it three cycles running, ordering by
    # float noise below the fourth decimal. Spread, not level, is what says "the
    # ballot cannot tell these lines apart".
    spread = max(scores.values()) - min(scores.values())
    flat = spread <= float(getattr(cfg.fix, "line_ballot_flat_epsilon", 0.001))
    saturated_multi = len(sat) >= 2 or flat
    if flat:
        saturated_unique = False
    win = sat[0] if saturated_unique else max(scores, key=scores.get)
    gap = scores[win] - scores[incumbent]
    rivals = [v for k, v in scores.items() if k != incumbent]
    lead = scores[incumbent] - max(rivals) if rivals else 0.0
    out.update(winner_line=win + 1, gap=round(gap, 4),
               saturated_unique=saturated_unique,
               saturated_multi=saturated_multi)
    # SATURATED-MULTI IS NO SIGNAL AT ALL (rung-4000 autopsy, ranked fix #3).
    # `curriculum_framework` scored {456: 0.9996, 458: 0.9996, 460: 0.9996} —
    # three lines identical to four decimal places, on a question the model
    # demonstrably cannot answer, with the gold (459) not among them. Blocking
    # SELECTION on saturation was already right; letting it still REORDER was
    # not, because the reorder is decided by float noise below the fourth
    # decimal and option order is empirically what the 1B's answer tracks. A
    # ballot that says the same thing about every line says nothing.
    if saturated_multi:
        _why = ("flat within %.4f" % spread) if flat and len(sat) < 2 \
            else "saturated on %d lines" % len(sat)
        out["decision"] = "kept: %s (no signal)" % _why
        out["select_blocked_by"] = _why
        out["ballot_spread"] = round(spread, 6)
        # NAMED, so a saturated field can be counted instead of inferred from
        # prose (stacked-autopsy runner-up 3). A ballot that says the same thing
        # about every line returns NO SIGNAL: no reorder, no select.
        out["ballot_saturated"] = True
        return out
    # margin the SELECTION gate weighs: a challenger's gap, or, when the ballot
    # confirms the incumbent, the incumbent's lead over its best rival. Both
    # drop the other lines, so both clear the same higher bar.
    margin = gap if win != incumbent else lead
    # SELECTION CLEARS A DOUBLE BAR (fix #3). Selection PRUNES — it deletes every
    # other line from the menu the model is about to read — and it did exactly
    # that to the gold line twice on this board: `book_series` dropped 1 line and
    # left a single option, `teacher_personas` dropped 3 and left a plausible
    # decoy the model then argued for correctly. A prune is unrecoverable, so it
    # now needs strictly more than `factor x select_tau`, and `saturated_unique`
    # no longer BYPASSES that bar — a lone 0.999 with a 0.09 lead is a confident
    # score, not a confident decision. Reordering is unchanged: the losing line
    # stays on the menu, which is what makes it cheap to be wrong about.
    select_bar = factor * sel_tau
    select_ok = can_select and margin > select_bar
    if not select_ok:
        out["select_blocked_by"] = (
            "select disabled" if not can_select
            else "margin %.4f below %.1fx select_tau (%.3f)"
                 % (margin, factor, select_bar))
    if win != incumbent:
        decisive = gap >= tau or saturated_unique
        if decisive and select_ok:
            out.update(decision="selected", winner=win)
        elif decisive:
            out.update(decision="reordered", winner=win)
        else:
            out["decision"] = "kept: ballot not decisive"
        return out
    out["incumbent_lead"] = round(lead, 4)
    # AGREEMENT, RECORDED, NO ACTION. The ballot's winner IS the line the menu
    # already leads with. There is nothing to reorder and nothing to prune, but
    # "the two independent signals agree" is a fact worth being able to count —
    # it is the difference between a thin margin nobody checked and a thin margin
    # that two mechanisms landed on together.
    out["ballot_agreement"] = True
    if select_ok:
        out.update(decision="selected: incumbent confirmed", winner=incumbent)
    else:
        out["decision"] = "kept: winner already leads"
    return out


# CANDIDATE-PROBE VERDICT — pure, so the selector's decision can be REPLAYED
# against a recorded cert exactly like `ballot_verdict` above. Every knob is read
# here; nothing else ranks.
#
# CODE RANKS (E49). The model never sees the candidates side by side and never
# names one: it answers N independent yes/no questions, and THIS function sorts
# the answers. Ranking key is `(-score, position)` — the incoming order is the
# evidence ranking the upstream stages produced (locate -> inert filter ->
# witness replay -> spec_exec shortlist -> line ballot), and it is used ONLY to
# break an exact score tie. It can therefore never restore the A-lock: a
# candidate at index 0 with a lower probe score loses to one at index 7.
#
# THREE HONEST ABSTAINS, none of which the letter menu could express:
#   all negative  the model affirmed no candidate at all -> nothing to select
#   flat          every candidate scored within `flat_epsilon` -> no signal
#   no scores     fewer than one readable probe -> the selector never ran
# The autopsy explicitly warns the sign is not to be trusted blindly (the line
# ballot scored gold at -0.96 on `ea639fc1`), which is why every per-candidate
# score is recorded in the cert for calibration and why the downstream apply
# gates still arbitrate. What the sign IS trusted for is refusing to select from
# a field the model rejected wholesale — that direction only ever costs a fix we
# had no evidence for.
def candidate_probe_verdict(scores, order, cfg):
    """{decision, winner, top, runner_up, margin, spread} — pure.

    scores  {candidate_position: signed score}, from `_yes_no_score`
    order   candidate positions in INCOMING (evidence-ranked) order; tie-break
    """
    eps = float(getattr(cfg.fix, "binary_probe_flat_epsilon", 0.001))
    out = {"winner": None, "flat_epsilon": eps,
           "scored": len(scores), "candidate_order": list(order)}
    if not scores:
        out["decision"] = "abstain: no probe scores"
        return out
    rank = {p: i for i, p in enumerate(order)}
    ranked = sorted(scores, key=lambda p: (-scores[p], rank.get(p, len(order))))
    top = ranked[0]
    spread = max(scores.values()) - min(scores.values())
    out["spread"] = round(spread, 6)
    out["top"] = round(scores[top], 4)
    if len(scores) > 1:
        out["runner_up"] = round(scores[ranked[1]], 4)
        out["margin"] = round(scores[top] - scores[ranked[1]], 4)
    else:
        out["runner_up"] = None
        # A single readable probe has no rival to beat. Its own signed score IS
        # the margin: a lone +0.9 is a real (if unopposed) affirmation, a lone
        # -0.9 is caught by the all-negative gate below.
        out["margin"] = round(scores[top], 4)
    # FLAT FIELD = NO SIGNAL (same measurement as ballot_verdict's flat gate).
    # Checked BEFORE the sign gate: a field flat at +0.999 is exactly as
    # uninformative as one flat at -0.999, and picking its argmax means ordering
    # by float noise below the fourth decimal.
    if len(scores) > 1 and spread <= eps:
        out["decision"] = "abstain: flat within %.4f (no signal)" % spread
        out["flat"] = True
        return out
    if scores[top] <= 0.0:
        out["decision"] = "abstain: no candidate affirmed"
        out["all_negative"] = True
        return out
    out.update(decision="ranked", winner=top)
    return out


class Fixer:
    """One run's state. Every knob read comes off self.cfg — no hardcoded behavior."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.margins = []
        # FAIL-CLOSED PARSING: set whenever a menu answer could not be read from
        # the declared {"pick": "X"} shape and the substring fallback was used.
        # The fallback is kept (behavior unchanged) but the certificate must say so.
        self.parse_fallback = False
        # per-call parse telemetry, read by menu_pick right after _one_menu
        self._last_parse_by = None
        self._last_raw = None
        self._last_reask_raw = None
        self._last_finish_reason = ""
        self._last_slot_clamped = False
        self._last_continuation = False
        self._last_failed_call = False
        self._last_usage = None
        self._last_inadmissible_request = None
        # re-entrancy guard for majority voting: the no-quorum fallthrough calls
        # menu_pick again and must land on the ordinary single-call path.
        self._voting = True
        # ---- BOUNDED INFO-REQUEST state (per TASK; see the header block) -----
        # `_offered_kinds` is set by the request wrapper for the duration of one
        # menu and is what switches action parsing on in `_one_menu`. Empty ⇒
        # every path below is byte-identical to the pre-request harness.
        self._offered_kinds = ()
        self._last_request = None
        self._req_active = False            # re-entrancy guard (mirrors _voting)
        self.request_ledger = []            # one row per ADMITTED request
        # Echoes and other unusable asks: NOT charged, but counted, because
        # "the syntax is un-elicitable" is only visible if the misses are kept.
        self.inadmissible_requests = []
        self.request_signatures = set()     # canonical dedup, task-wide
        self.requests_spent_task = 0
        self.request_budget_per_menu = int(
            getattr(cfg.fix, "info_request_budget_per_menu", 1))
        self.request_budget_task = int(
            getattr(cfg.fix, "info_request_budget_per_task", 2))
        self.uledger = None                 # lazily built; fail-open
        self.request_events = []            # DecisionEvent dicts
        # RETRY-SKIP predicate (autopsy fix #1a). None = no witness gate applies
        # to this menu; run() installs one for the mutation menu only.
        self.witness_admits = None

    # -------------------------------------------------------------- menus
    def _reask_one_letter(self, prompt, prose, letters, label=None):
        """RE-ASK (autopsy fix #2, fallback half). The model answered in PROSE —
        it may well have answered CORRECTLY in prose — and the harness recorded
        an ABSTAIN. Hand its own words back and ask for the letter alone, with a
        budget too small to write another paragraph.

        Never fires on an empty answer: that is a transport failure, not a
        formatting one, and re-asking it would charge a call for nothing. Never
        yields a letter the menu did not offer (`_lenient_letter` is in-range by
        construction). Returns (letter | None, raw_text).
        """
        if not (prose or "").strip():
            return None, ""
        try:
            reask_prompt = REASK_Q.format(
                menu_prompt=prompt, prose=_prompt_json(prose, 600),
                letters="/".join(letters))
            text, _resp = _call(
                reask_prompt, seed=self.cfg.seed, url=self.cfg.url,
                max_tokens=budget_for_stage(self.cfg, "reask", reask_prompt),
                label=(label or "") + ":reask",
                grammar=pick_grammar(letters)
                if (getattr(self.cfg.kernel, "grammar_pick", False)
                    and not stage_thinking(self.cfg, "reask")) else None,
                enable_thinking=stage_thinking(self.cfg, "reask"),
                continuation=CONTINUATION_REASK_Q)
        except Exception:
            return None, ""             # fail OPEN: the menu still abstains
        return _lenient_letter(text, letters), (text or "")[:200]

    def _one_menu(self, build_prompt, options, displays, label=None):
        letters = [chr(65 + i) for i in range(len(options))]
        menu = "\n".join(f"  {letters[i]}) {d}" for i, d in enumerate(displays))
        prompt = build_prompt(menu)
        n_probs = 8 if self.cfg.kernel.margin_probe else 0
        configure_traces(self.cfg)      # honor kernel.capture_traces per run
        grammar = (pick_grammar(letters)
                   if getattr(self.cfg.kernel, "grammar_pick", False) else None)
        text, resp = _call(prompt, seed=self.cfg.seed, url=self.cfg.url,
                           max_tokens=budget_for_stage(self.cfg, "menu", prompt),
                           n_probs=n_probs, label=label, grammar=grammar,
                           enable_thinking=stage_thinking(self.cfg, "menu"),
                           continuation=CONTINUATION_Q)
        self._last_trace_ref = last_trace_ref()
        # #11d(c): a guillotined answer and a real refusal both arrive as
        # pick=None. Persisting finish_reason keeps them forever distinguishable
        # in the ledger, so "abstained" can never again mean "ran out of room".
        self._last_finish_reason = _finish_reason(resp)
        _meta = call_meta()
        self._last_slot_clamped = _meta["slot_clamped"]
        self._last_continuation = _meta["continuation_used"]
        self._last_usage = _meta["usage"]
        # FAILED CALL, NOT A VOTE (stacked-autopsy half 1c): truncated with
        # nothing to read even after the continuation re-ask. The quorum layer
        # excludes these instead of scoring them as dissent.
        self._last_failed_call = bool(
            self._last_finish_reason == "length" and not (text or "").strip())
        margin = _letter_margin(resp, letters) if n_probs else None
        # Self-diagnosis: logprobs came back but no letter token matched. Persist
        # the head of the token stream so the NEXT live run says why, instead of
        # leaving a bare margin:null to be re-investigated by hand.
        debug = None
        if n_probs and margin is None:
            head = _token_strings(resp)
            if head:
                debug = head
        self._last_parse_by = None
        self._last_raw = (text or "")[:200]
        # ---- ONE ACTION PER REPLY. The action parse runs BEFORE every pick
        # parser on purpose: `REQUEST(SHOW_SOURCE,A)` contains the letter A and
        # the substring/lenient rescues below would happily read it as a pick.
        # Inert unless the request wrapper armed `_offered_kinds` for this menu.
        self._last_request = None
        self._last_inadmissible_request = None
        if self._offered_kinds:
            act = parse_action(text, letters, self._offered_kinds)
            if act is not None and act["action"] == "request":
                if act.get("inadmissible"):
                    # Ignored as a non-action: not charged, does not abort the
                    # ballot, does not close the channel. Recorded so the
                    # elicitability of the syntax stays measurable.
                    self._last_inadmissible_request = act
                    self.inadmissible_requests.append(
                        {"menu": label, "kind": act.get("kind"),
                         "targets": list(act.get("targets") or ()),
                         "error": act.get("error", ""),
                         "raw": act.get("raw", "")[:120]})
                else:
                    self._last_request = act
                    self._last_parse_by = "request"
                    return None, margin, debug
            if act is not None and act["action"] == "abstain":
                self._last_parse_by = "abstain"
                return None, margin, debug
        m = re.search(r'"pick"\s*:\s*"([A-Z])"', text or "")
        if m and m.group(1) in letters:
            self._last_parse_by = "strict"
            return options[letters.index(m.group(1))], margin, debug
        hits = [o for o in options if str(o) in (text or "")]
        candidate_text = (text or "").strip().strip("`\"'")
        if (len(set(map(str, hits))) == 1
                and candidate_text == str(hits[0]).strip()):
            # A bare candidate path/name is deterministic; prose that merely
            # mentions a candidate is not a decision and must not be rescued.
            self.parse_fallback = True
            self._last_parse_by = "candidate_text"
            return hits[0], margin, debug
        # LENIENT LADDER (last resort, never a guess between two letters)
        letter = _lenient_letter(text, letters)
        if letter is not None:
            self._last_parse_by = "lenient"
            return options[letters.index(letter)], margin, debug
        # Natural-language decisions are common in the captured answer slot;
        # accept one only when its decision phrase is unambiguous.
        letter = _prose_letter(text, letters)
        if letter is not None:
            self._last_parse_by = "prose"
            return options[letters.index(letter)], margin, debug
        # ONE-LETTER RE-ASK (autopsy fix #2): a non-empty PROSE answer is a
        # formatting failure, not a refusal. Ask once more for the letter alone.
        if getattr(self.cfg.kernel, "grammar_reask", False) and (text or "").strip():
            letter2, raw2 = self._reask_one_letter(prompt, text, letters,
                                                   label=label)
            self._last_reask_raw = raw2
            if letter2 is not None:
                self._last_parse_by = "reask"
                return options[letters.index(letter2)], margin, debug
        return None, margin, debug

    def _menu_pick_voted(self, build_prompt, options, displays, label, votes):
        """MAJORITY VOTING over within-boot nondeterminism (kernel.menu_votes).

        MEASURED PROBLEM: 7 of 21 gym tasks flip pass/fail across IDENTICAL runs
        — same boot, same seed, same prompt. The server's continuous batching
        reorders float reductions, so a near-tie menu decision lands on either
        side of the tie depending on who else is in the batch. The prompt here is
        therefore IDENTICAL on every vote ON PURPOSE: the batching jitter IS the
        sampler, and perturbing the prompt would measure something else.

        Resolution, strictly ordered:
          unanimous     every vote cast, all agreeing      -> that pick, N/N
          majority      a strict majority of cast votes    -> that pick, k/N
          no majority   the votes disagree without a lead  -> the EXISTING
                        thin-margin reversed-retry/abstain path, which is the
                        correct answer for a genuine coin flip
        Ordering matters (a): voting resolves FIRST, and the reversed retry fires
        only when voting could not decide. Reversing a menu that three votes
        already agreed on buys nothing and can only introduce a disagreement.

        PARSE FAILURES ARE NON-VOTES (b), never a vote for None: an unreadable
        answer is an absence of evidence, and counting it as a ballot for
        "abstain" would let a formatting bug outvote the model. If MORE THAN HALF
        the votes fail to parse there is no quorum, so the round falls through to
        the ordinary single-call path (which re-asks and runs its own ladder) and
        the row records `vote_parse_failures`.

        BUDGET (d): menu calls have no allocator budget to overdraw. The
        attention allocator sets exactly two run-local values —
        `line_ballot_max_calls` and `fix_retries` — and menu_pick charges
        NEITHER, so votes cannot spend either one. A cap attribute nothing
        writes was deliberately NOT added: a reader with no writer is the
        dead-feature shape this repo keeps re-learning. Every vote's cost stays
        auditable in the row instead.
        """
        picks, vmargins, vparse, vraw, vfinish = [], [], [], [], []
        vfailed, vclamped, vcont, vcompletion = [], False, False, []
        debug0, tref0, fb_any = None, None, False
        for i in range(votes):
            _fb_before = self.parse_fallback
            p, m, d = self._one_menu(build_prompt, options, displays, label=label)
            fb_any = fb_any or (self.parse_fallback and not _fb_before)
            # VOTING x INFO-REQUEST INTERACTION (documented contract): votes are
            # cast on the FINAL, post-request menu state. So the moment ANY vote
            # asks for evidence, this round is abandoned mid-ballot — the votes
            # already cast were cast against a menu that is about to change and
            # counting them would mix two different questions. The wrapper
            # fulfils the request and RESTARTS the votes against the augmented
            # menu, under the SAME per-menu budget: the restart grants no new
            # request budget, so the runaway bound (budget + 1 rounds) holds
            # with voting exactly as it does without it.
            if self._last_request is not None:
                self.margins.append({
                    "menu": label, "n": len(options), "pick": None,
                    # `margin: None` EXPLICITLY. Every other margins record
                    # carries the key, and the DecisionEvent round-trip reads it
                    # by subscript because it is one of "the fields the uledger
                    # reads" — a record that silently omits it KeyErrors the
                    # reconstruction the moment a real run happens to abort a
                    # ballot for an info request. Measured 2026-08-29: the live
                    # `ea639fc1` cert did exactly that on its `line` menu and
                    # took `test_roundtrips_the_fields_the_uledger_reads` down.
                    # A menu that never finished voting HAS no margin, and that
                    # is a value, not an absent key (#223: one spelling, always
                    # present, on both sides of the seam).
                    "margin": None,
                    "n_votes": votes, "votes_cast_before_request": i,
                    "vote_outcome": "aborted_for_info_request"})
                return None
            picks.append(p)
            vmargins.append(m)
            vparse.append(self._last_parse_by)
            # only keep raw text for the votes that FAILED — that is the half a
            # later autopsy cannot reconstruct
            vraw.append(None if self._last_parse_by else self._last_raw)
            vfinish.append(getattr(self, "_last_finish_reason", ""))
            vfailed.append(bool(getattr(self, "_last_failed_call", False)))
            vclamped = vclamped or bool(getattr(self, "_last_slot_clamped", False))
            vcont = vcont or bool(getattr(self, "_last_continuation", False))
            _vu = getattr(self, "_last_usage", None)
            if isinstance(_vu, dict) and _vu.get("completion_tokens") is not None:
                vcompletion.append(_vu["completion_tokens"])
            if i == 0:
                debug0, tref0 = d, getattr(self, "_last_trace_ref", None)
        cast = [p for p in picks if p is not None]
        failed_votes = sum(1 for f in vfailed if f)
        # ATTEMPTED vs RETURNED (stacked-autopsy half 2). A vote that never came
        # back is a MISSING MEASUREMENT, not a dissent, and it must never be
        # allowed to outvote a parsed one. Three of last cycle's four cowardly
        # abstains had exactly one real vote and it was CORRECT.
        attempted = votes
        returned = len(cast)
        failures = votes - returned
        tally = {}
        for p in cast:
            key = str(p)[:80]
            tally[key] = tally.get(key, 0) + 1

        def _annotate(rec):
            rec["n_votes"] = votes
            rec["votes"] = tally
            rec["vote_picks"] = [str(p)[:80] if p is not None else None
                                 for p in picks]
            rec["vote_margins"] = vmargins
            rec["vote_parse_by"] = vparse
            rec["vote_finish_reasons"] = vfinish
            rec["quorum_basis"] = {"returned": returned, "attempted": attempted,
                                   "failed_votes": failed_votes}
            if failed_votes:
                rec["failed_votes"] = failed_votes
            if vclamped:
                rec["slot_clamped"] = True
            if vcont:
                rec["continuation_used"] = True
            if failures:
                rec["vote_parse_failures"] = failures
                rec["vote_raw_responses"] = vraw
            if vcompletion:
                rec["completion_tokens"] = sum(vcompletion)
            return rec

        # NO QUORUM: NOTHING came back. Only a completely silent round has
        # nothing to say — one parsed vote out of three attempts is unanimous
        # among the evidence that exists, and it now decides. (Previously
        # `failures * 2 > votes` let two silences outvote one real vote; that
        # rule alone produced 4 of the 12 failures on `3de6826bbea7`.)
        if returned == 0:
            self._voting = False
            try:
                pick = self.menu_pick(build_prompt, options, displays, label)
            finally:
                self._voting = True
            if self.margins:
                _annotate(self.margins[-1])["vote_outcome"] = "no_quorum"
            return pick

        top = max(tally.values())
        winners = [k for k, v in tally.items() if v == top]
        # UNANIMOUS AMONG RETURNED VOTES. The old `len(cast) == votes` half made
        # unanimity a property of the SEATS; a single dropped call demoted a
        # clean 2/2 agreement to "no_majority" and sent it to the reversal.
        unanimous = len(tally) == 1
        # FULL unanimity — every SEAT returned and agreed — stays a separate,
        # stricter fact, because the retry-skip rule is the one place where the
        # count of silences is genuinely relevant: skipping the position-bias
        # reversal on a menu that only half answered would retire that guard on
        # a technicality. Resolution uses `unanimous`; the skip uses this.
        unanimous_full = unanimous and returned == attempted
        majority = len(winners) == 1 and top * 2 > returned
        pick = None
        if unanimous or majority:
            win = winners[0]
            pick = next(p for p in cast if str(p)[:80] == win)

        rec = {"menu": label, "n": len(options), "margin": vmargins[0],
               "options": [str(o)[:80] for o in options],
               "pick": str(pick) if pick is not None else None,
               "finish_reason": vfinish[0]}
        _annotate(rec)
        rec["vote_outcome"] = ("unanimous" if unanimous_full else
                               "majority" if (unanimous or majority)
                               else "no_majority")
        # DENOMINATOR IS RETURNED VOTES, not seats: "1/1" is the honest record of
        # one vote that came back and decided, and `quorum_basis` carries the
        # attempted count so the silence is never hidden.
        rec["vote_margin"] = "%d/%d" % (top, returned)
        if "length" in vfinish:
            rec["truncated"] = True
        if debug0:
            rec["margin_debug"] = debug0
        if tref0:
            rec["trace_ref"] = {"file": tref0["file"], "line": tref0["line"],
                                "index_base": tref0.get("index_base", 0),
                                "offset": tref0.get("offset")}
            rec["trace_tokens"] = tref0["tokens"]
            rec["trace_shape"] = tref0.get("shape")
        if fb_any:
            rec["parse_fallback"] = True
        if vparse[0]:
            rec["parse_by"] = vparse[0]
        elif pick is None:
            rec["raw_response"] = vraw[0]

        # REVERSED RETRY — fires AFTER voting, on either of two triggers:
        #   no_majority  the votes disagreed without a lead: treat it exactly as
        #                the thin-margin case and let the reversal break the tie
        #   thin_margin  the votes AGREED but the first vote's top-2 logprob gap
        #                is still under tau
        # Keeping the second trigger is load-bearing and was nearly lost here.
        # Voting and the margin probe catch DIFFERENT failures: voting samples
        # batching jitter, while the reversal catches POSITION BIAS — a model
        # answering "A" because it is first. N identical calls agree with each
        # other by construction in that case, so voting alone would have retired
        # the position-bias guard without a single test noticing. Two did.
        # The thin test reads the MINIMUM margin across the votes, not vote #0's.
        # Keying it on the first vote alone was a real defect caught by a test:
        # an unparseable first vote carries margin=None, which silently switched
        # the position-bias guard off for the whole menu. Min is also the
        # conservative choice — if ANY vote was fragile, verify.
        k = self.cfg.kernel
        _seen = [m for m in vmargins if m is not None]
        _mmin = min(_seen) if _seen else None
        if _mmin is not None and _mmin != vmargins[0]:
            rec["margin_min"] = _mmin
        thin = k.margin_probe and _mmin is not None and _mmin < k.margin_tau
        # ---- RETRY SKIP (arbitration rule (a), rung-4000 autopsy fix #1) ----
        # A UNANIMOUS pre-retry vote whose winner also survived the witness gate
        # is not a coin flip, and the reversal has nothing left to measure: the
        # two tasks this fired on (`utils/image_processing.py`, unanimous 3/3 on
        # the gold FILE; `api/code_submissions.py`, unanimous 3/3 on the
        # byte-exact gold MUTATION) were both converted into abstains by a
        # position-swap disagreement. The retry is not free either — it is a
        # whole extra call whose only possible effect on a unanimous menu was to
        # veto it. Witness admission is what keeps this from rubber-stamping the
        # `none of these` escape (math_answer_verifier's unanimous-but-wrong
        # vote): 'none' is never witness-admitted, so a unanimous abstention
        # still gets challenged.
        if (thin and pick is not None and unanimous_full
                and getattr(k, "retry_skip_on_unanimous", True)
                and self._witness_admits(pick)):
            rec["arbitration"] = {
                "rule_applied": "unanimous_survivor_kept",
                "retry_called": False,
                "pre_retry": str(pick)[:80],
                "vote_margin": rec["vote_margin"],
                "why": "unanimous pre-retry vote whose pick survived the witness "
                       "gate; the reversal can only veto, never inform"}
            rec["resolved"] = str(pick)
            self.margins.append(rec)
            return pick
        if pick is None or thin:
            incumbent = pick if pick is not None else (cast[0] if cast else None)
            rec["retry"] = "reversed"
            rec["retry_reason"] = "no_majority" if pick is None else "thin_margin"
            r_opts = list(reversed(options))
            r_disp = list(reversed(displays))
            pick2, margin2, debug2 = self._one_menu(build_prompt, r_opts, r_disp,
                                                    label=(label or "") + ":reversed")
            _tref2 = getattr(self, "_last_trace_ref", None)
            if _tref2:
                rec["trace_ref_retry"] = {
                    "file": _tref2["file"], "line": _tref2["line"],
                    "index_base": _tref2.get("index_base", 0),
                    "offset": _tref2.get("offset")}
                rec["trace_tokens_retry"] = _tref2["tokens"]
            if self._last_parse_by:
                rec["parse_by_retry"] = self._last_parse_by
            elif pick2 is None:
                rec["raw_response_retry"] = self._last_raw
            if debug2:
                rec["margin_debug_retry"] = debug2
            rec["margin_retry"] = margin2
            rec["pick_retry"] = str(pick2) if pick2 is not None else None
            pick, arb = self._arbitrate(incumbent, pick2, cast,
                                        unanimous or majority, rec["vote_margin"])
            rec["arbitration"] = arb
            if pick is None:
                rec["abstain"] = True
            rec["pick"] = str(pick) if pick is not None else None
            rec["resolved"] = str(pick) if pick is not None else None
        self.margins.append(rec)
        return pick

    # ------------------------------------------------- retry arbitration (#1)
    def _witness_admits(self, pick):
        """Did the witness gate positively admit this option?

        `run()` installs `self.witness_admits` (a predicate over menu options)
        for the mutation menu once witness replay has spoken. It is None on every
        menu with no witness gate to fail — file and function menus — and a gate
        that does not apply must not be readable as a gate that FAILED, so None
        admits. `('none', None, None)` is never admitted: the escape option is
        not a candidate code ever replayed.
        """
        try:
            if isinstance(pick, tuple) and pick and pick[0] == "none":
                return False
            fn = getattr(self, "witness_admits", None)
            if fn is None:
                return True
            return bool(fn(pick))
        except Exception:                # a broken predicate never decides
            return False

    def _arbitrate(self, incumbent, retry_pick, cast, had_majority, vote_margin):
        """THE RETRY IS A TIE-BREAK, NOT A VETO (rung-4000 autopsy, ranked #1).

        MEASURED: the reversed-order probe fired 6 times on the failing half of
        the board, rescued 0 failures and converted 4 would-be gold contacts into
        abstains. In FIVE of six firings the correct answer was present on one
        side of the retry and `disagreement -> abstain` threw it away.

        Disagreement between two orderings of the same menu is evidence about
        POSITION SENSITIVITY, not about which option is right. So it now resolves
        by weight of evidence instead of by veto:

          agreed              both sides name the same option          -> it wins
          retry_unreadable    the reversal produced no pick at all     -> the
                              pre-retry decision stands if it had a majority;
                              an unreadable challenger is an absence of evidence
          both_sides          the retry's option ALSO drew a pre-retry vote ->
                              that option wins, whatever the tally said. This is
                              `tools/grading.py`: 2/3 voted `none of these`, the
                              third vote AND the reversal both named the gold.
          disjoint_majority   the two sides name options with no overlap -> the
                              pre-retry MAJORITY stands: it carries 2-3 votes,
                              the reversal carries one.
          no_majority         the pre-retry votes never agreed AND the reversal
                              disagrees with all of them -> ABSTAIN. This is the
                              only surviving veto, and it is the genuine coin
                              flip the probe was built for.

        Returns `(pick, {rule_applied, ...})`. The rule is recorded on the margin
        row so a later autopsy can count each branch instead of inferring it.
        """
        arb = {"retry_called": True,
               "pre_retry": str(incumbent)[:80] if incumbent is not None else None,
               "retry_pick": str(retry_pick)[:80] if retry_pick is not None else None,
               "vote_margin": vote_margin}
        if not getattr(self.cfg.kernel, "retry_tiebreak", True):
            # legacy veto, kept switchable so the A/B is one knob wide
            agreed = (incumbent is not None and retry_pick is not None
                      and str(incumbent) == str(retry_pick))
            arb["rule_applied"] = "agreed" if agreed else "legacy_veto_abstain"
            return (incumbent if agreed else None), arb
        if incumbent is not None and retry_pick is not None \
                and str(incumbent) == str(retry_pick):
            arb["rule_applied"] = "agreed"
            return incumbent, arb
        if retry_pick is None:
            arb["rule_applied"] = ("retry_unreadable_kept_majority" if had_majority
                                   else "retry_unreadable_no_majority_abstain")
            return (incumbent if had_majority else None), arb
        both = [p for p in cast if str(p) == str(retry_pick)]
        if both:
            arb["rule_applied"] = "both_sides"
            arb["pre_retry_votes_for_retry_pick"] = len(both)
            return both[0], arb
        if had_majority and incumbent is not None:
            arb["rule_applied"] = "disjoint_kept_majority"
            return incumbent, arb
        arb["rule_applied"] = "no_majority_disagreement_abstain"
        return None, arb

    # ------------------------------------------------- bounded info requests
    def _charge_request(self, req, label, fulfilled, latency, detail=""):
        """ADMISSION accounting. Called for EVERY request the parser produced —
        malformed, duplicate, unfulfillable or fulfilled — before any fetch is
        allowed to matter. There is no refund path, by design."""
        self.requests_spent_task += 1
        sig = req.get("signature") or request_signature(
            req.get("kind") or "?", req.get("targets") or [], req.get("args") or {})
        row = {"menu": label, "kind": req.get("kind"),
               "targets": list(req.get("targets") or ()),
               "args": dict(req.get("args") or {}),
               "signature": sig, "fulfilled": bool(fulfilled),
               "latency_ms": int(latency * 1000),
               "error": req.get("error") or detail or "",
               "task_spent": self.requests_spent_task,
               "raw": req.get("raw", "")[:120]}
        self.request_signatures.add(sig)
        self.request_ledger.append(row)
        self._ledger_write(row)
        return row

    def _ledger_write(self, row):
        """U-ledger row + DecisionEvent per request. FAIL-OPEN: the request
        mechanism must never be taken down by its own bookkeeping."""
        try:
            from detreason import uledger as _ul, decision_event as _de
            if self.uledger is None:
                self.uledger = _ul.ULedger("fixer:info_requests")
            rid = "req-%02d" % len(self.request_ledger)
            urow = _ul.URow(
                id=rid, utype=_ul.UNVERIFIED_CLAIM,
                claim="menu %s could not be decided without %s%s" % (
                    row["menu"], row["kind"], tuple(row["targets"])),
                severity=0.4, u=1.0, source="fixer.info_request",
                owner="fixer")
            self.uledger.open(urow)
            if row["fulfilled"]:
                self.uledger.close(rid, _ul.ClosureRecord(
                    criterion=_ul.EVIDENCE_SPAN, actor="fixer.fetch",
                    evidence_ids=[row["signature"]],
                    detail={"span": [0, 1], "source": row["kind"]}))
            row["urow_id"] = rid
            ev = _de.DecisionEvent(
                correlation_id="fixer:info_requests",
                producer=_de.Producer(module="fixer.info_request", version="1",
                                      bundle_fingerprint=self.cfg.fingerprint()),
                decision_type=_de.SENSOR_READ,
                choice=row["signature"],
                u_rows_opened=(rid,),
                u_rows_closed=((rid,) if row["fulfilled"] else ()),
                outcome=("fulfilled" if row["fulfilled"] else
                         (row["error"] or "unfulfillable")),
                context={"kind": "info_request", "menu": row["menu"]})
            self.request_events.append(ev.to_dict())
        except Exception as _e:          # bookkeeping is never load-bearing
            row.setdefault("ledger_error", type(_e).__name__)

    def _menu_with_requests(self, build_prompt, options, displays, label,
                            rctx, kinds):
        """The BOUNDED loop. Worst case is `budget + 1` decision rounds and that
        is the whole runaway-train guarantee — every exit below either returns a
        decision or spends budget, and nothing inside can add budget back."""
        letters = [chr(65 + i) for i in range(len(options))]
        cap_lines = int(getattr(self.cfg.fix, "info_request_context_lines", 12))
        cap_chars = int(getattr(self.cfg.fix, "info_request_evidence_max_chars", 1200))
        remaining_task = max(0, self.request_budget_task - self.requests_spent_task)
        budget = min(self.request_budget_per_menu, remaining_task)
        blocks, made, spent = [], [], 0
        pick = None
        while True:
            closed = spent >= budget
            offered = () if closed else kinds

            _brace = bool(getattr(self.cfg.facts, "brace_sanitize", False))

            def _bp(menu, _b=tuple(blocks), _o=offered, _s=_brace):
                return (build_prompt(menu)
                        + render_evidence(list(_b), cap_chars, brace_sanitize=_s)
                        + request_affordance(_o, letters))

            self._offered_kinds = kinds     # detection stays on even when CLOSED,
            self._req_active = True         # so a stray request cannot be read as
            try:                            # a pick by the lenient ladder
                pick = self.menu_pick(_bp, options, displays, label)
            finally:
                self._req_active = False
                self._offered_kinds = ()
            req = self._last_request
            self._last_request = None
            if req is None:
                break                       # a pick or an honest abstain: done
            if closed:
                # DETERMINISTIC CONVERSION: the affordance no longer offers
                # REQUEST, so this is a stray. It becomes an abstain, never
                # another round.
                made.append({"kind": req.get("kind"), "targets": req.get("targets"),
                             "outcome": "converted_to_abstain",
                             "charged": False})
                pick = None
                break
            t0 = time.time()
            sig = req.get("signature")
            got, detail = None, ""
            if req.get("error"):
                detail = req["error"]
            elif sig in self.request_signatures:
                detail = "duplicate request signature"
            else:
                try:
                    got = fulfill_request(req, rctx, options, letters, cap_lines)
                except Exception as _e:
                    got, detail = None, "fetch failed (%s)" % type(_e).__name__
                if got is None and not detail:
                    detail = "unfulfillable"
            row = self._charge_request(req, label, got is not None,
                                       time.time() - t0, detail)
            spent += 1
            made.append(row)
            if got is not None:
                blocks.append({"kind": req["kind"], "targets": req["targets"],
                               **got})
            # NOTE: no `budget` mutation anywhere in this loop. A
            # request-triggered re-ask grants no new budget — that is the
            # difference between a bounded mechanism and a runaway train.
        _inadm = [r for r in self.inadmissible_requests if r["menu"] == label]
        if _inadm:
            _rec = self.margins[-1] if self.margins else None
            if isinstance(_rec, dict) and _rec.get("menu") == label:
                _rec["inadmissible_requests"] = _inadm
        if made:
            rec = self.margins[-1] if self.margins else None
            if isinstance(rec, dict) and rec.get("menu") == label:
                rec["info_requests"] = made
                rec["info_request_budget"] = {
                    "per_menu": self.request_budget_per_menu,
                    "per_task": self.request_budget_task,
                    "spent_this_menu": spent,
                    "spent_this_task": self.requests_spent_task,
                    "requests_closed": spent >= budget}
                # CONVERSION is the number the whole feature lives or dies on:
                # did asking for evidence actually produce a decision?
                rec["request_to_pick"] = bool(
                    pick is not None and any(m.get("fulfilled") for m in made))
        return pick

    def menu_pick_binary(self, options, displays, label, probe_q,
                         confirm_q=None, is_candidate=None,
                         probe_continuation=CONTINUATION_BINARY_Q,
                         confirm_continuation=CONTINUATION_BINARY_Q):
        """SELECTOR REPLACEMENT for the letter menu (conversion autopsy §3).

        Asks ONE independent yes/no question per candidate over its rendered
        `old -> new` edit, ranks the answers in CODE (`candidate_probe_verdict`),
        then spends ONE call letting the model confirm or VETO the top. Returns
        the winning option, or None on an honest abstain.

        Position-free BY CONSTRUCTION: each probe is a separate call that sees
        exactly one candidate, so there is no menu order for the answer to track.
        The incoming order is still recorded in the certificate — it is the
        tie-break, and a reader has to be able to see what it was.

        BUDGET: N probes + 1 confirm, N capped at `fix.binary_probe_max`.
        `probe_q(opt)` / `confirm_q(opt)` build the prompts; `is_candidate(opt)`
        excludes non-edits (the 'none of these' sentinel), which are never probed
        because there is nothing to render and nothing to affirm.
        """
        if not options:
            return None
        configure_traces(self.cfg)
        cap = max(0, int(getattr(self.cfg.fix, "binary_probe_max", 12)))
        keep = is_candidate or (lambda _o: True)
        # positions are indices into `options`, so a score is always traceable
        # back to the exact candidate that earned it
        cand_pos = [i for i, o in enumerate(options) if keep(o)]
        capped = len(cand_pos) > cap
        probe_pos = cand_pos[:cap]
        scale = float(getattr(self.cfg.fix, "binary_probe_margin_scale", 4.0))
        requested_scoring = str(getattr(self.cfg.fix, "probe_scoring",
                                        "logit_margin") or "logit_margin").lower()
        scoring = (requested_scoring if requested_scoring in ("token",
                                                              "logit_margin")
                   else "logit_margin")
        scores, token_scores, logit_scores, raw_margins = {}, {}, {}, {}
        candidate_scores, fallback_reasons = {}, {}
        calls, unreadable, how, trefs = 0, 0, {}, {}
        for i in probe_pos:
            try:
                probe_prompt = probe_q(options[i])
                _t, _r = _call(probe_prompt, seed=self.cfg.seed,
                               url=self.cfg.url, n_probs=20,
                               max_tokens=budget_for_stage(
                                   self.cfg, "binary_probe", probe_prompt),
                               enable_thinking=stage_thinking(
                                   self.cfg, "binary_probe"),
                               label=(label or "") + ":probe",
                               continuation=probe_continuation)
                calls += 1
                token_score, logit_score, raw_margin, status = _yes_no_scores(
                    _r, scale)
                # PER-PROBE trace ref. Each probe is its own call, so there is no
                # single trace for this row — and inheriting `self._last_trace_ref`
                # from the PREVIOUS menu would attribute another decision's
                # reasoning to this one, which is worse than recording nothing.
                _tr = last_trace_ref()
                if _tr:
                    trefs[str(i)] = {"file": _tr["file"], "line": _tr["line"],
                                     "index_base": _tr.get("index_base", 0),
                                     "offset": _tr.get("offset")}
            except Exception:
                token_score = logit_score = raw_margin = None
                status = "call failed"  # fail OPEN: a dead probe abstains
            token_scores[str(i)] = token_score
            logit_scores[str(i)] = logit_score
            raw_margins[str(i)] = raw_margin
            if scoring == "token":
                s = token_score
                scored_by = "token"
            elif logit_score is not None:
                s = logit_score
                scored_by = "logit_margin"
            else:
                s = token_score
                scored_by = "token_fallback"
                fallback_reasons[str(i)] = status
            how[str(i)] = scored_by
            candidate_scores[str(i)] = {
                "token_score": (None if token_score is None
                                 else round(token_score, 4)),
                "logit_margin": (None if raw_margin is None
                                  else round(raw_margin, 4)),
                "logit_margin_score": (None if logit_score is None
                                        else round(logit_score, 4)),
                "selected_score": None if s is None else round(s, 4),
                "scored_by": scored_by,
            }
            if s is None:
                unreadable += 1
            else:
                scores[i] = s
        verdict = candidate_probe_verdict(scores, probe_pos, self.cfg)
        win = verdict.get("winner")
        probe = {"selector": "binary", "cap": cap, "capped": capped,
                 "candidates": len(cand_pos), "probed": len(probe_pos),
                 "calls": calls, "unreadable": unreadable,
                 "probe_scoring": scoring,
                 "probe_scoring_requested": requested_scoring,
                 # PER-CANDIDATE SCORES, always. The autopsy asks for these by
                 # name: the ballot scored gold NEGATIVE on `ea639fc1`, so the
                 # sign is calibration data, not something to trust silently.
                 "scores": {str(k): round(v, 4) for k, v in scores.items()},
                 # Keep both measurements for calibration, even when one is not
                 # the active selector. `None` means the server did not expose a
                 # usable YES/NO class pair at that answer position.
                 "token_scores": {
                     k: (None if v is None else round(v, 4))
                     for k, v in token_scores.items()},
                 "logit_margin_scores": {
                     k: (None if v is None else round(v, 4))
                     for k, v in logit_scores.items()},
                 "logit_margins": {
                     k: (None if v is None else round(v, 4))
                     for k, v in raw_margins.items()},
                 "candidate_scores": candidate_scores,
                 # DIAGNOSTICS the calibration pass needs: which scorer
                 # actually answered each probe, and the length bucket the score
                 # was earned in. A score whose provenance is unknown cannot be
                 # pooled with one that is.
                 "scored_by": how,
                 "margin_scale": scale,
                 "length_chars": {str(i): len(str(displays[i]))
                               for i in probe_pos if i < len(displays)},
                 "displays": {str(i): str(displays[i])[:80]
                               for i in probe_pos if i < len(displays)}}
        if fallback_reasons:
            # LOUD FALLBACK: a default logit-margin run that silently degrades to
            # sampled-token scoring is exactly the calibration failure this knob
            # exists to expose.
            probe["logit_margin_fallback"] = True
            probe["logit_margin_fallback_candidates"] = sorted(
                fallback_reasons, key=lambda value: int(value))
            probe["logit_margin_fallback_reasons"] = fallback_reasons
        probe.update(verdict)
        # SATURATION, added 2026-08-29 after the file-stage regression. A field
        # where nearly every candidate is affirmed is not a ranking — on
        # `ea639fc1` six of eight candidates scored >= 0.6 and the gold file lost
        # to a known confabulation attractor by 0.0195. `margin` alone did not
        # make that legible, because a large spread (one candidate at -1.0) sat
        # next to a photo-finish at the top. These two numbers together do.
        # DIAGNOSTIC ONLY — nothing gates on them; a selector that silently
        # abstained on saturation would trade a wrong pick for no pick and
        # report neither.
        if scores:
            probe["affirmed"] = sum(1 for v in scores.values() if v > 0.0)
            probe["affirmed_frac"] = round(
                probe["affirmed"] / float(len(scores)), 3)
        pick = None
        if win is not None:
            pick = options[win]
            if confirm_q is not None and getattr(
                    self.cfg.fix, "binary_probe_confirm", True):
                try:
                    confirm_prompt = confirm_q(pick)
                    _t, _r = _call(confirm_prompt, seed=self.cfg.seed,
                                   url=self.cfg.url, n_probs=20,
                                   max_tokens=budget_for_stage(
                                       self.cfg, "binary_probe", confirm_prompt),
                                   enable_thinking=stage_thinking(
                                       self.cfg, "binary_probe"),
                                   label=(label or "") + ":confirm",
                                   continuation=confirm_continuation)
                    calls += 1
                    confirm_token, confirm_logit, confirm_margin, confirm_status = (
                        _yes_no_scores(_r, scale))
                    if scoring == "token":
                        raw_cs, chow = confirm_token, "token"
                    elif confirm_logit is not None:
                        raw_cs, chow = confirm_logit, "logit_margin"
                    else:
                        raw_cs, chow = confirm_token, "token_fallback"
                        probe["confirm_logit_margin_fallback"] = True
                        probe["confirm_logit_margin_fallback_reason"] = (
                            confirm_status)
                    probe["confirm_token_score"] = (
                        None if confirm_token is None else round(confirm_token, 4))
                    probe["confirm_logit_margin"] = (
                        None if confirm_margin is None else round(confirm_margin, 4))
                except Exception:
                    raw_cs, chow = None, "call failed"
                    probe["confirm_token_score"] = None
                    probe["confirm_logit_margin"] = None
                # POLARITY FLIP. CAND_CONFIRM_Q asks the INVERSE question ("can
                # the problem still happen?"), so a YES is a VETO. Negating here
                # keeps one convention everywhere below: higher = stronger
                # evidence the candidate addresses the problem.
                cs = None if raw_cs is None else -raw_cs
                probe["confirm_calls"] = 1
                probe["confirm_inverse_question"] = True
                probe["confirm_raw_score"] = (None if raw_cs is None
                                              else round(raw_cs, 4))
                probe["confirm_scored_by"] = chow
                probe["confirm_score"] = None if cs is None else round(cs, 4)
                # PROPORTIONALITY, added 2026-08-29. The veto used to be a bare
                # sign check (`cs < 0.0`), so a confirm score of -0.0068 could
                # overturn a winner probed at +0.875 — one unrepeated call at
                # the noise floor outranking the whole ranked field. That is a
                # soft claim beating hard evidence (#197), and it was the single
                # gate behind three ABSTAINs of the ref_partial2 cycle
                # (2dbb4371, fb1d75cb, 97062f14 — the last with a 0.546 margin
                # over its runner-up).
                #
                # THE GATE IS NOT WEAKENED WHERE IT WAS RIGHT. The six rows that
                # vetoed on real evidence (|cs| 0.43 - 0.997) all clear the
                # threshold and still veto; see `binary_probe_confirm_ratio` in
                # gym_config.py for the measured distribution the 0.25 sits in.
                # An unreadable confirm still fails OPEN, exactly as before.
                _cr = float(getattr(self.cfg.fix,
                                    "binary_probe_confirm_ratio", 0.0) or 0.0)
                _wscore = verdict.get("top")
                _floor = (_cr * max(0.0, float(_wscore))
                          if isinstance(_wscore, (int, float)) else 0.0)
                probe["confirm_veto_ratio"] = _cr
                probe["confirm_veto_floor"] = round(_floor, 4)
                if cs is not None and cs < 0.0 and -cs < _floor:
                    # DECLINED VETO — recorded, never silent. The winner stands
                    # on its own probe evidence and the contradicting-but-weak
                    # confirm travels with it so a reader can second-guess this.
                    probe["confirm_veto_declined"] = {
                        "confirm_score": round(cs, 4),
                        "winner_score": _wscore,
                        "floor": round(_floor, 4)}
                    probe["confirmed"] = False
                elif cs is not None and cs < 0.0:
                    # VETO. The model was shown the winner alone and said no.
                    probe["decision"] = "abstain: confirm vetoed winner"
                    probe["confirmed"] = False
                    probe["vetoed_winner"] = win
                    # `winner` must agree with `pick`, or a cert reader counting
                    # winners counts a selection that never happened. The ranked
                    # candidate is not lost — it is `vetoed_winner`.
                    probe["winner"] = None
                    pick, win = None, None
                else:
                    # fail OPEN on an unreadable confirm — recorded, not silent
                    probe["confirmed"] = cs is not None
                    if cs is None:
                        probe["confirm_unreadable"] = True
            probe["calls"] = calls
        rec = {"menu": label, "n": len(options),
               "margin": verdict.get("margin"),
               "options": [str(o)[:80] for o in options],
               "pick": str(pick) if pick is not None else None,
               "parse_by": "binary_probe",
               "selector": "binary",
               "binary_probe": probe}
        if trefs:
            # NOT `rec["trace_ref"]`: that key means "the one call behind this
            # decision", and this decision has N of them.
            probe["trace_refs"] = trefs
        self.margins.append(rec)
        return pick

    def menu_pick_pairwise(self, options, displays, label, pair_q):
        """CHAMPION TOURNAMENT over an evidence-ranked candidate list.

        file-probe design §4, implemented with his controls. The incoming
        order is the CODE RANKER's, and it is the thing being challenged: the
        champion starts as candidate 0 and a challenger only takes the crown by
        WINNING a comparison. A TIE keeps the champion, so an undecided model
        leaves the ranker's answer exactly where it was — which is the correct
        failure mode when the alternative is a coin flip.

        COST: N-1 comparisons, plus ONE position-swapped rematch when the model
        has displaced the ranker's leader. Compare that with N probes + a confirm
        for the absolute selector: this is cheaper AND asks a question a 1B can
        actually answer.

        POSITION CONTROL, which is the whole reason this exists: orientation
        alternates deterministically by round, so a model that always answers "A"
        wins and loses in equal measure instead of riding slot order. The final
        rematch is run with the orientation REVERSED from the bout that won the
        crown; a winner that cannot repeat itself with the sides swapped is
        UNRESOLVED, and unresolved means the ranker keeps its leader — never a
        forced winner, never an abstain we could have avoided.
        """
        if not options:
            return None
        configure_traces(self.cfg)
        n = len(options)
        rounds, calls = [], 0

        def _compare(i, j, flip):
            """Ask about candidates i and j. `flip` puts j on the A side."""
            a, b = (j, i) if flip else (i, j)
            prompt = pair_q(options[a], displays[a], options[b], displays[b])
            try:
                _t, _r = _call(prompt, seed=self.cfg.seed, url=self.cfg.url,
                               max_tokens=budget_for_stage(
                                   self.cfg, "menu", prompt),
                               enable_thinking=stage_thinking(
                                   self.cfg, "menu"),
                               label=(label or "") + ":pair",
                               continuation=CONTINUATION_Q)
            except Exception:
                return None, "call failed"
            tok = parse_pairwise_answer(_t)
            if tok is None:
                return None, "unparseable"
            if tok.upper() == "TIE":
                return "tie", tok
            # map the SIDE back to the candidate index, undoing the flip
            return (a if tok == "A" else b), tok

        champion = 0
        for k in range(1, n):
            flip = bool(k % 2)          # alternate which side the champion sits
            won, how = _compare(champion, k, flip)
            calls += 1
            rounds.append({"champion": champion, "challenger": k,
                           "champion_side": "B" if flip else "A",
                           "answer": how,
                           "winner": None if won in (None, "tie") else won})
            if won not in (None, "tie") and won != champion:
                champion = won
        rec = {"menu": label, "n": n, "selector": "pairwise",
               "parse_by": "pairwise_tournament",
               # a tournament has no logprob margin — that is a VALUE, not a
               # missing key (see the aborted-ballot record above)
               "margin": None,
               "options": [str(o)[:80] for o in options],
               "pairwise": {"rounds": rounds, "calls": calls,
                            "ranker_leader": 0, "champion": champion}}
        pick = options[champion]
        if champion != 0:
            # THE MODEL DISPLACED THE RANKER. This is the one result that
            # overturns deterministic evidence, so it is the one that gets a
            # control: the challenger must beat the ranker's leader in BOTH
            # orientations.
            #
            # MEASURED WHY BOTH, 2026-08-29: a single swapped rematch is not
            # enough, and the first version of this method shipped that bug. It
            # chose the rematch orientation from "the last round the champion
            # won", which includes rounds the champion merely DEFENDED — and an
            # A-locked model defends from the A side, so the mirror was run in
            # the very orientation that flatters it. The offline A-lock fixture
            # crowned index 3. Requiring both orientations is the only form of
            # this control that a side-preferring model cannot pass: it wins
            # exactly one of the two, whichever side it prefers.
            mirror = []
            for _flip in (False, True):
                _w, _how = _compare(0, champion, _flip)
                calls += 1
                mirror.append({"answer": _how, "champion_side":
                               "A" if _flip else "B",
                               "winner": None if _w in (None, "tie") else _w})
            rec["pairwise"]["calls"] = calls
            rec["pairwise"]["rematch"] = mirror
            if not all(m["winner"] == champion for m in mirror):
                # UNRESOLVED. The ranker's leader stands — we do not abstain,
                # because deterministic evidence is still evidence and it was
                # never the thing that failed here.
                rec["pairwise"]["unresolved"] = True
                rec["pairwise"]["champion"] = 0
                pick = options[0]
        rec["pick"] = str(pick)
        self.margins.append(rec)
        return pick

    def menu_pick(self, build_prompt, options, displays, label, rctx=None):
        """A menu decision + its margin probe. Low margin -> re-ask with the
        candidate order REVERSED (answer mapped back); disagreement -> abstain.

        With kernel.menu_votes > 1 the SAME menu is asked N times first and the
        majority decides — see _menu_pick_voted. menu_votes=1 leaves everything
        below byte-identical to the pre-voting path."""
        if not options:
            return None
        # BOUNDED INFO-REQUEST wrapper. Only fires when the CALL SITE supplied a
        # RequestContext (so a fetch is actually possible), the knob is on, and
        # this menu is in the offer table. Everything below is untouched.
        if rctx is not None and not self._req_active:
            _kinds = menu_request_kinds(label, self.cfg)
            if _kinds:
                return self._menu_with_requests(build_prompt, options, displays,
                                                label, rctx, _kinds)
        _votes = int(getattr(self.cfg.kernel, "menu_votes", 1) or 1)
        if _votes > 1 and len(options) > 1 and getattr(self, "_voting", True):
            # a one-option menu has nothing to vote about; don't burn N calls
            return self._menu_pick_voted(build_prompt, options, displays,
                                         label, _votes)
        _fb_before = self.parse_fallback
        pick, margin, margin_debug = self._one_menu(build_prompt, options,
                                                    displays, label=label)
        _menu_used_fallback = self.parse_fallback and not _fb_before
        # A request is not a thin-margin abstain: return it to the wrapper
        # UNSPENT rather than burning the reversed-order retry on a menu the
        # model has explicitly said it cannot yet answer.
        if self._last_request is not None:
            self.margins.append({
                "menu": label, "n": len(options), "margin": margin,
                "options": [str(o)[:80] for o in options], "pick": None,
                "parse_by": "request",
                "finish_reason": getattr(self, "_last_finish_reason", "")})
            return None
        _parse_by, _raw = self._last_parse_by, self._last_raw
        _finish = getattr(self, "_last_finish_reason", "")
        # mechanical observability: persist MEMBERSHIP so diagnosis can split
        # recall-vs-pick exactly (the diagnosis engine's #1 instrumentation ask)
        rec = {"menu": label, "n": len(options), "margin": margin,
               "options": [str(o)[:80] for o in options],
               "pick": str(pick) if pick is not None else None,
               "finish_reason": _finish}
        if _finish == "length":
            # survived the ladder and STILL truncated: this is a budget failure
            # the run must not be able to dress up as a considered refusal.
            rec["truncated"] = True
        if getattr(self, "_last_slot_clamped", False):
            rec["slot_clamped"] = True
        if getattr(self, "_last_continuation", False):
            rec["continuation_used"] = True
        if getattr(self, "_last_failed_call", False):
            rec["failed_call"] = True
        _usage = getattr(self, "_last_usage", None)
        if isinstance(_usage, dict) and _usage.get("completion_tokens") is not None:
            rec["completion_tokens"] = _usage["completion_tokens"]
        if margin_debug:
            rec["margin_debug"] = margin_debug
        # THINKING TRACE: ref only. The body lives in results/traces/<key>.jsonl
        # — a cert carrying full traces is unreadable and unbounded.
        _tref = getattr(self, "_last_trace_ref", None)
        if _tref:
            # `index_base` travels WITH the ref: a bare {file, line} was read as
            # 1-based by a downstream autopsy and mis-attributed every trace.
            rec["trace_ref"] = {"file": _tref["file"], "line": _tref["line"],
                                "index_base": _tref.get("index_base", 0),
                                "offset": _tref.get("offset")}
            rec["trace_tokens"] = _tref["tokens"]
            rec["trace_shape"] = _tref.get("shape")
        if _menu_used_fallback:
            rec["parse_fallback"] = True
        if _parse_by:
            rec["parse_by"] = _parse_by
        elif pick is None:
            # UNDIAGNOSABLE-ABSTAIN FIX: an unparseable answer used to leave only
            # `pick: null`. Persist the raw head so the next run says WHY.
            rec["raw_response"] = _raw
        k = self.cfg.kernel
        if k.margin_probe and margin is not None and margin < k.margin_tau:
            rec["retry"] = "reversed"
            r_opts = list(reversed(options))
            r_disp = list(reversed(displays))
            pick2, margin2, debug2 = self._one_menu(build_prompt, r_opts, r_disp,
                                                    label=(label or "") + ":reversed")
            _tref2 = getattr(self, "_last_trace_ref", None)
            if _tref2:
                rec["trace_ref_retry"] = {
                    "file": _tref2["file"], "line": _tref2["line"],
                    "index_base": _tref2.get("index_base", 0),
                    "offset": _tref2.get("offset")}
                rec["trace_tokens_retry"] = _tref2["tokens"]
            if self._last_parse_by:
                rec["parse_by_retry"] = self._last_parse_by
            elif pick2 is None:
                rec["raw_response_retry"] = self._last_raw
            if debug2:
                rec["margin_debug_retry"] = debug2
            rec["margin_retry"] = margin2
            rec["pick_retry"] = str(pick2) if pick2 is not None else None
            # SINGLE-CALL PATH: there is exactly one pre-retry answer, so there is
            # no majority for the tie-break rules of `_arbitrate` to weigh — this
            # IS the no-majority branch, and it keeps the veto. The rule is still
            # recorded so the ledger names every arbitration outcome (a branch
            # with no label is a branch nobody can count).
            pick, arb = self._arbitrate(pick, pick2,
                                        [pick] if pick is not None else [],
                                        False, "1/1")
            rec["arbitration"] = arb
            if pick is None:
                rec["abstain"] = True
            rec["pick"] = str(pick) if pick is not None else None
            rec["resolved"] = str(pick) if pick is not None else None
        self.margins.append(rec)
        return pick

    # ------------------------------------------------------------ locate
    def class_pattern_rank(self, cls, all_files):
        """LOCATE HAMMER stage (a_disjoint_locate #2): the symptom CLASS implies
        code patterns; rank files by class-pattern density x size-normalization.
        Pure C0 — recovers vocabulary-disjoint files that share zero tokens with
        the symptom but are dense in the implicated construct."""
        pats = {
            "cmp_flip": r" >= | <= | > | < ",
            "bool_flip": r" and | or ",
            "none_flip": r" is None| is not None",
            "arith_flip": r"\w [+\-] \w",
        }
        pat = pats.get(cls)
        if not pat:
            return []
        scored = []
        for path, src in all_files.items():
            lines = src.splitlines()
            hits = len(re.findall(pat, src))
            if not hits:
                continue
            # WINDOWED density (dilution fix): max class-pattern density over
            # sliding 30-line windows — a dense function inside a big file ranks
            # itself instead of being drowned by file size.
            best_w = 0
            per_line = [len(re.findall(pat, l)) for l in lines]
            w = sum(per_line[:30])
            best_w = w
            for i in range(30, len(per_line)):
                w += per_line[i] - per_line[i - 30]
                if w > best_w:
                    best_w = w
            density = hits / max(1, len(lines))
            scored.append((best_w, density, hits, path))
        # UNION of two views (recall-first: windowed count found the dense block
        # in a big file; whole-file density finds the small dense file — choosing
        # one traded misses for misses, measured):
        by_window = sorted(scored, key=lambda s: -s[0])
        by_density = sorted(scored, key=lambda s: -s[1])
        out = []
        for a, b in zip(by_window, by_density):
            for s in (a, b):
                if s[3] not in out:
                    out.append(s[3])
        return out[:24]

    def cascade(self, task_text, idx, df, n, adj, log, cls=None, all_files=None):
        """g1 core, knob-routed: BM25 top-K UNION class-pattern-rank UNION
        import-proximity -> pool cap -> 1B relevance atoms (reject confident NOs
        only) -> menu_k. Recall-first: sources union, never serially filter."""
        c = self.cfg.locate
        ranked = bm25_rank(task_text, idx, df, n)
        lex = [p for _, p in ranked[:c.bm25_top]]
        seeds = lex[:c.prox_seed]
        prox = set(seeds)
        for p in seeds:
            prox |= adj.get(p, set())
        clsrank = self.class_pattern_rank(cls, all_files) if all_files else []
        log["cls_pattern_candidates"] = clsrank[:8]
        # Cycle-2 lesson (fixed 2->1, contact 8->6): naive interleave let noisy
        # class-pattern candidates OUTRANK lexical hits and poisoned menu order.
        # Class candidates join the POOL (atoms sift them) but lexical order
        # leads; class-only candidates append after. Recall kept, precision back.
        pool = list(dict.fromkeys(lex + clsrank + sorted(prox)))[:c.pool_cap]
        if not c.relevance_atoms:
            return pool[:c.menu_k]
        kept = []
        for p in pool:
            idents = ", ".join(sorted(idx[p], key=lambda w: -idx[p][w])[:12])
            relevance_prompt = REL_Q.format(task=task_text, path=p,
                                            idents=idents)
            text, _ = _call(
                relevance_prompt, seed=self.cfg.seed, url=self.cfg.url,
                max_tokens=budget_for_stage(self.cfg, "relevance",
                                             relevance_prompt),
                enable_thinking=stage_thinking(self.cfg, "relevance"),
                label="locate:relevance",
                continuation=CONTINUATION_RELEVANCE_Q)
            out = text or ""
            ans = parse_relevance_answer(out) or "unsure"
            if ans != "no":
                kept.append(p)
            log.setdefault("rel_atoms", []).append({p: ans})
        if not kept:
            kept = pool
        order = {p: i for i, p in enumerate(pool)}
        kept.sort(key=lambda p: order[p])
        return kept[:c.menu_k]


def load_repo(root, stats=None):
    """Collect the candidate corpus. `stats` (optional dict) receives
    `skipped_unreadable` — an unreadable file is a silenced input, never a
    vanished one (fail-open must not be fail-silent)."""
    root = pathlib.Path(root)
    files = {}
    skipped = 0
    for p in sorted(root.rglob("*.py")):
        if "test" in p.name or "migration" in str(p) or "/." in str(p):
            continue
        try:
            files[str(p.relative_to(root))] = p.read_text()
        except Exception:
            skipped += 1
    if stats is not None:
        stats["skipped_unreadable"] = skipped
    return root, files


# ------------------------------------------------------------- safe apply
def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _resolve_inside(root, rel_path):
    """Resolve `rel_path` under `root` and prove it stays inside.

    Returns the resolved path, or raises ValueError. Rejects symlinks outright
    (a symlinked target is a write to somewhere else by definition) and any
    resolved path whose parts do not start with the resolved root's parts.
    """
    root_r = pathlib.Path(root).resolve()
    target = (root_r / rel_path)
    if target.is_symlink() or any(
            parent.is_symlink() for parent in target.parents
            if str(parent).startswith(str(root_r))):
        raise ValueError("apply: refusing to write through a symlink")
    target_r = target.resolve()
    if target_r.parts[:len(root_r.parts)] != root_r.parts:
        raise ValueError("apply: resolved path escapes the repo root")
    if target_r.is_symlink():
        raise ValueError("apply: refusing to write through a symlink")
    return root_r, target_r


def _atomic_write(target, text):
    """Write via a temp file in the SAME directory + os.replace (atomic rename)."""
    d = target.parent
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".fixer-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _comment_facts(cfg, path, src, fnpick, stats=None):
    """Optional typed comment channel. The module may not exist yet — absence is
    fine and must never break a run. Every swallowed exception is COUNTED into
    `stats["facts_errors"]` so a permanently-dead channel is visible."""
    def _err():
        if stats is not None:
            stats["facts_errors"] = stats.get("facts_errors", 0) + 1
    if not cfg.facts.comments_as_facts:
        return []
    if not GIT_ROOT:
        raise RuntimeError("comment facts require STATEFRAME_GIT_ROOT")
    try:
        from comment_facts import extract_comment_facts  # noqa: F401
    except ImportError:
        return []
    try:
        try:
            out = extract_comment_facts(
                src=src, path=path, fn=fnpick, git_root=GIT_ROOT,
                stale_months=cfg.facts.comment_stale_months)
        except TypeError:
            out = extract_comment_facts(src, fnpick, git_root=GIT_ROOT)
    except Exception:
        _err()
        return []
    if isinstance(out, str):
        return [out]
    try:
        return [str(x) for x in out]
    except TypeError:
        _err()
        return []


def _strict_hole_answer(raw, hole):
    """Decode one complete typed-hole answer without positional salvage."""
    text = str(raw or "").strip()
    if text.lower().startswith("<think>"):
        end = text.lower().find("</think>")
        if end < 0:
            return None
        text = text[end + len("</think>"):].strip()
    if (not text or "\n" in text or "\r" in text
            or "```" in text or "<think>" in text.lower()
            or "</think>" in text.lower()):
        return None
    if text[:1] in "\"'" or text[-1:] in "\"'":
        if len(text) < 2 or text[0] != text[-1] or text[0] not in "\"'":
            return None
        text = text[1:-1]
    if not text:
        return None
    kind = getattr(hole, "kind", "")
    if kind == "string_value":
        if re.match(r"(?i)^(?:answer|response|result|output|value|proposal|"
                    r"explanation|reason)\s*:", text):
            return None
        if re.match(r"(?i)^(?:the\s+)?(?:answer|response|result|value)"
                    r"\s+(?:is|means)\s+", text):
            return None
        if re.match(r"(?i)^(?:i\s+think|here\s+is|think\s*[—:-])\b", text):
            return None
        return text
    if kind == "fstring_text":
        return text
    if not (text.isascii() and text.isidentifier()):
        return None
    return text


def _install_strict_hole_decoder(module):
    """Make the imported hole pipeline use the fixer protocol at its boundary."""
    decoder = getattr(module, "_decode", None)
    if not callable(decoder) or getattr(decoder, "_fixer_strict", False):
        return

    def strict_decoder(raw, hole):
        return _strict_hole_answer(raw, hole)

    strict_decoder._fixer_strict = True
    module._decode = strict_decoder


def hole_completer(cfg, label="hole_fill"):
    """The model seam for a typed-hole fill with a bounded answer budget.

    Two deliberate choices, both measured lessons:

    * `budget_for` sizes the request, never the raw token count. #11d: this 1B
      spends ~350 tokens THINKING before it emits anything, and llama.cpp counts
      that inside the same budget — a site that asks for 16 "answer" tokens gets
      16 tokens TOTAL and a truncated non-answer.
    * The grammar is passed ONLY when `cfg.fix.typed_hole_grammar` is on, and it
      is OFF by default, because `KernelCfg.grammar_pick` records the live
      measurement: GBNF is incompatible with thinking mode on this build and
      every constrained call came back EMPTY. `holes.validate_fill` is the
      authority either way, so the grammar is an optimisation we are not
      currently entitled to claim.
    """
    # The seam's keyword is `answer_tokens`, NOT `max_tokens`, and deliberately:
    # `max_tokens` is the thing llama.cpp counts THINKING against, and naming a
    # parameter after it invites exactly the #11d defect of handing a raw answer
    # size to the server. What a caller passes here is how many tokens the
    # ANSWER needs; `budget_for` adds the reasoning headroom.
    def complete(prompt, grammar=None, answer_tokens=None, seed=0):
        g = grammar if getattr(cfg.fix, "typed_hole_grammar", False) else None
        text, _raw = _call(
            prompt, seed=seed, url=cfg.url,
            max_tokens=budget_for_stage(cfg, "hole", prompt,
                                        answer_tokens=answer_tokens), n_probs=0,
            label=label, grammar=g,
            enable_thinking=stage_thinking(cfg, "hole"),
            continuation=CONTINUATION_FILL_Q)
        return text
    return complete


def _typed_hole_tranche(span, lo, lines, base_muts, cfg, cert, complete=None):
    """Enumerate typed-hole templates, fill them, return telemetry.

    The candidates ride out on the ``_candidates`` key so the caller decides
    whether to append them — this function never mutates the menu itself.

    FAIL-OPEN, NEVER FAIL-SILENT. Any failure (no templates, a dead provider, an
    exhausted budget, a validator that rejected every sample) yields zero
    candidates and a telemetry record that NAMES the cause. An empty tranche
    reported as ``{"emitted": 0}`` with no reason would be indistinguishable
    from "this function has no typed holes", which is the silent-drop class.
    """
    tel = {"enabled": True, "candidates": 0, "_candidates": []}
    try:
        import holes as _holes
        _install_strict_hole_decoder(_holes)
    except Exception as exc:                              # pragma: no cover
        tel["error"] = "import:%s" % type(exc).__name__
        return tel
    try:
        templates, meta = _holes.enumerate_hole_templates(
            span, lo, lines, base_candidates=base_muts,
            cap=getattr(cfg.fix, "typed_hole_templates",
                        _holes.MAX_HOLE_TEMPLATES))
    except Exception as exc:                              # law 1
        tel["error"] = "enumerate:%s" % type(exc).__name__
        return tel
    tel["templates"] = meta
    if not templates:
        return tel
    if complete is None:
        complete = hole_completer(cfg)
    budget = _holes.FillBudget(getattr(cfg.fix, "typed_hole_fill_calls",
                                       _holes.MAX_FILL_CALLS))
    try:
        cands, ftel = _holes.fill_templates(
            templates, span, lo, lines, complete=complete,
            samples=getattr(cfg.fix, "typed_hole_samples",
                            _holes.FILL_SAMPLES),
            budget=budget)
    except Exception as exc:                              # law 1
        tel["error"] = "fill:%s" % type(exc).__name__
        tel.update(budget.telemetry())
        return tel
    for c in cands:
        try:
            c.from_hole = True        # so the prose/inert filters can REPORT
        except AttributeError:        # pragma: no cover
            pass
    tel.update(ftel)
    tel["_candidates"] = cands
    return tel


# ======================================================== BLOCK-HOLE FAMILY
# Wired 2026-08-29 (the wiring lane). `block_holes.BLOCK_HOLE_SEAM` names the
# contract; everything below is the fixer half of it and nothing here reaches
# into block_holes' internals.

def block_completer(cfg, label="block_hole"):
    """The model seam for a block-hole authoring call with a bounded answer budget.

    Byte-for-byte the same conventions as `hole_completer`, and deliberately so:
    the keyword is `answer_tokens` (NOT `max_tokens`) because llama.cpp counts
    THINKING against the server budget (#11d), and `budget_for` is what adds the
    reasoning headroom. A block answer is ~320 tokens rather than a leaf's 16,
    so getting this wrong here truncates a whole block instead of one name.

    No grammar is ever passed. `holes`' measurement applies unchanged — GBNF is
    incompatible with thinking mode on this build and every constrained call came
    back EMPTY — and `block_holes.validate_block` is the authority regardless.
    """
    def complete(prompt, answer_tokens=None, seed=0, **_kw):
        text, _raw = _call(
            prompt, seed=seed, url=cfg.url,
            max_tokens=budget_for_stage(cfg, "block", prompt,
                                        answer_tokens=answer_tokens), n_probs=0,
            label=label, enable_thinking=stage_thinking(cfg, "block"),
            continuation=CONTINUATION_BLOCK_Q)
        return text
    return complete


_BLOCK_ANSWER_TOKENS = 320   # mirrors block_holes.ANSWER_TOKENS; kept local so
                             # importing the module is not a hard dependency of
                             # merely constructing the completer.

#: ROLE GRAMMAR for the menu/probe card. A block candidate's raw diff
#: must NOT go into the 1B menu prompt: it is many lines, it dwarfs every
#: single-line option next to it, and option length is a known bias channel.
#: What goes in instead is one deterministic sentence generated from the ROLE
#: and the SITE — never from the model, never from a comment.
#:
#: `{subject}` is the site's own symptom-overlapping subject when it has one and
#: the empty string otherwise; a slot we cannot fill from canonical metadata
#: degrades the phrase to its role-only form rather than inventing a filler.
_BLOCK_ROLE_GRAMMAR = {
    "guard":      "exit early when {subject} is unsafe, before line {line}",
    "sanitize":   "normalize {subject} before it is used, at line {line}",
    "revalidate": "re-check {subject} after the barrier and stop if invalid, "
                  "at line {line}",
    "loop_guard": "skip or stop the iteration when {subject} holds, at line "
                  "{line}",
    "other":      "write the missing statements at line {line}",
}


def block_probe_phrase(role, site, span_start=0, subjects=()):
    """ONE deterministic sentence describing a block candidate. Presentation only.

    It cannot upgrade evidence and it cannot change ranking: a phrase is
    what the menu SHOWS, and the candidate is still arbitrated on its patch by
    spec_exec, witness replay and the gates, exactly as an enumerated one is.
    """
    subject = ""
    for s in (subjects or ()):
        if isinstance(s, str) and s.isidentifier():
            subject = s
            break
    tpl = _BLOCK_ROLE_GRAMMAR.get(role or "other", _BLOCK_ROLE_GRAMMAR["other"])
    line = span_start + getattr(site, "line", 0) + 1
    verb = "replace" if getattr(site, "kind", "insert") == "replace" else "insert"
    body = tpl.format(subject=(subject or "the value"), line=line)
    return "%s a block: %s" % (verb, body)


def _block_hole_tranche(span, lo, lines, cfg, cert, symptom="", fn_name=None,
                        complete=None):
    """Enumerate block-hole sites, author + validate blocks, return telemetry.

    Same discipline as `_typed_hole_tranche`, and for the same reasons:

    * candidates ride out on ``_candidates`` so the CALLER decides whether to
      append them — this function never mutates the menu;
    * FAIL-OPEN, NEVER FAIL-SILENT. A dead provider, an exhausted budget, a span
      that will not parse or a validator that refused every sample all yield
      zero candidates AND a telemetry record that NAMES the cause. `{"emitted":
      0}` with no reason is indistinguishable from "this function has no legal
      block site", which is the silent-drop class (#181c).

    MENU PRESSURE is applied HERE rather than downstream, because it is
    a property of this family and not of the menu:

        raw frontier   <= site_limit x samples   (the seam's own bound)
        screen         <= block_hole_screen  (8) deduped by canonical AST
        expose         <= block_hole_expose  (2) into the menu

    Screening is a DEDUPE plus a cap, never a quality judgement — this lane owns
    no witness of its own, and using one candidate's shape to evict another's
    would be a ranking dressed as a filter. The full frontier is reported in the
    certificate so an eviction is auditable.
    """
    tel = {"enabled": True, "candidates": 0, "_candidates": []}
    try:
        import block_holes as _bh
    except Exception as exc:                              # pragma: no cover
        tel["error"] = "import:%s" % type(exc).__name__
        return tel
    try:
        import holes as _holes
    except Exception as exc:                              # pragma: no cover
        tel["error"] = "import:%s" % type(exc).__name__
        return tel
    if complete is None:
        complete = block_completer(cfg)
    budget = _holes.FillBudget(getattr(cfg.fix, "block_hole_fill_calls",
                                       _BLOCK_FILL_CALLS))
    site_tel = []
    try:
        raw = _bh.block_hole_candidates(
            span, lo, lines, symptom=symptom, complete=complete,
            fn_name=fn_name,
            samples=getattr(cfg.fix, "block_hole_samples", _bh.BLOCK_SAMPLES),
            site_limit=getattr(cfg.fix, "block_hole_sites", _bh.MAX_SITES),
            budget=budget,
            # The KNOB is the authority here, not the module default.
            # `BLOCK_HOLE_ENABLED` is False in block_holes.py on purpose — that
            # module refuses to turn itself on — so passing `enabled` explicitly
            # is what makes the fixer's own default the one that decides.
            enabled=True,
            telemetry=site_tel)
    except Exception as exc:                              # law 1
        tel["error"] = "author:%s" % type(exc).__name__
        tel.update(budget.telemetry())
        return tel
    tel.update(budget.telemetry())
    # Per-site rejection tallies — a family that produced nothing because every
    # sample hit REJECT_UNKNOWN_NAME is a different fact from a family whose
    # provider was dead, and the referendum needs to tell them apart.
    rejected = {}
    calls_failed = 0
    for _sid, _st in site_tel:
        calls_failed += int((_st or {}).get("call_failed", 0) or 0)
        for k, v in ((_st or {}).get("rejected") or {}).items():
            rejected[k] = rejected.get(k, 0) + int(v or 0)
    tel["sites"] = len(site_tel)
    tel["rejected"] = rejected
    tel["call_failed"] = calls_failed
    tel["site_telemetry"] = [
        {"site_id": sid, **(st or {})} for sid, st in site_tel
    ]
    tel["frontier"] = len(raw)
    # ---- SCREEN: dedupe by canonical AST, then cap ------------------------
    screen_cap = max(0, int(getattr(cfg.fix, "block_hole_screen", 8)))
    expose_cap = max(0, int(getattr(cfg.fix, "block_hole_expose", 2)))
    screened, seen = [], set()
    duped = 0
    for cand in raw:
        try:
            key = _bh.ast_fingerprint(cand[2])
        except Exception:                                 # law 1: fail OPEN
            key = None
        # An unfingerprintable candidate is kept under its literal text rather
        # than dropped: a dedupe that cannot see is not a licence to evict.
        key = key or ("text", cand[2])
        if key in seen:
            duped += 1
            continue
        seen.add(key)
        screened.append(cand)
        if len(screened) >= screen_cap:
            break
    tel["deduped"] = duped
    tel["screened"] = len(screened)
    exposed = screened[:expose_cap]
    tel["exposed"] = len(exposed)
    role = _bh.classify_role(symptom)
    tel["role"] = role
    # The probe phrase rides on the candidate so the menu renderer does not have
    # to know what a block is. Best-effort: a candidate that will not carry the
    # attribute simply renders as an ordinary flattened diff.
    _sites = {}
    try:
        _fn_node, _facts = _bh.target_facts(span, lines, fn_name)
        if _fn_node is not None:
            for s in _bh.enumerate_sites(
                    _fn_node, symptom,
                    limit=getattr(cfg.fix, "block_hole_sites", _bh.MAX_SITES)):
                _sites[s.site_id] = s
    except Exception:                                     # law 1
        _sites = {}
    for c in exposed:
        try:
            c.from_block_hole = True   # so the prose/inert filters can REPORT
            # `make_block_edit` stamps `pattern = "block_hole:<kind>:<site_id>"`.
            _sid = str(getattr(c, "pattern", "")).rsplit(":", 1)[-1]
            _site = _sites.get(_sid)
            if _site is not None:
                c.probe_phrase = block_probe_phrase(role, _site, lo,
                                                    _site.subjects)
        except AttributeError:                            # pragma: no cover
            pass
    tel["candidates"] = len(exposed)
    tel["_candidates"] = exposed
    return tel


_BLOCK_FILL_CALLS = 48


def run(root, task_text, apply=False, cfg: "gym_config.GymConfig | None" = None,
        target_file: "str | None" = None, run_id: "str | None" = None,
        task_id_value: "str | None" = None):
    """PUBLIC CONTRACT: run(root, task_text, apply=False) must keep working with
    no cfg (a bench is built against it). cfg is purely additive.

    target_file  when the CALLER already knows which file the edit belongs to
                 (a chain plan pins it), the locate cascade / file menu is
                 skipped entirely and that file is used. Purely additive:
                 target_file=None keeps the default locate behaviour exactly.

    run_id/task_id_value optionally provide stable correlation identifiers for
    prompt-ledger rows; omitted values are generated or derived locally.
    """
    cfg = cfg or gym_config.default()
    resolved_run_id = str(run_id or prompt_ledger.new_run_id())
    resolved_task_id = str(task_id_value or prompt_ledger.task_id(task_text))
    try:
        ledger_cfg = getattr(cfg, "prompt_ledger", None)
        prompt_ledger.configure(
            enabled=bool(getattr(ledger_cfg, "enabled", True)),
            root=getattr(ledger_cfg, "directory", None) or None,
            max_bytes=getattr(ledger_cfg, "max_bytes", None),
        )
    except Exception:
        # A diagnostics sink must never make a repair unavailable.
        pass
    try:
        prompt_ledger.set_context(resolved_run_id, resolved_task_id)
    except Exception:
        # Correlation metadata is useful, but a broken diagnostics sink is not
        # allowed to block the repair pipeline.
        pass
    fx = Fixer(cfg)
    t0 = time.time()
    # TOKEN THROUGHPUT (BY REFERENCE, same pattern as `margins` below): every
    # `_call` this thread makes for the rest of this run mutates this exact
    # dict, so every one of run()'s many early returns already carries the
    # right totals with no per-return-point wiring. Fail-soft throughout
    # (_record_usage) — a response with no usage block simply adds nothing.
    usage_totals = _new_usage_totals()
    set_usage_sink(usage_totals)
    truncation = _new_truncation_stats()
    set_truncation_sink(truncation)
    cert = dict(task=task_text[:200], run_id=resolved_run_id,
                task_id=resolved_task_id, stages=[], gates=[], verdict="ABSTAIN",
                margins=fx.margins, config_fingerprint=cfg.fingerprint(),
                config=cfg.to_dict(), usage_totals=usage_totals,
                truncation=truncation)
    io_stats = {"skipped_unreadable": 0, "facts_errors": 0}
    cert["io_stats"] = io_stats
    # BY REFERENCE on purpose: run() has ~10 early returns, and a request made
    # on the file menu must still reach the certificate of a task that abstained
    # two stages later. The lists are mutated in place by the Fixer.
    cert["info_requests"] = fx.request_ledger
    cert["info_request_events"] = fx.request_events
    # ---- ATTENTION ALLOCATOR (detreason/attention.py) ------------------------
    # ARMED 2026-08-29. The allocator was built, tested and deterministic, and
    # NOTHING read it — a registry entry with no grant path, which is the dead
    # -feature shape this repo keeps re-learning. It now runs once at task start
    # and its `per_stage` depths become THIS RUN's values for the two knobs the
    # allocator's stage vocabulary actually maps onto: BALLOT -> the line-ballot
    # call budget, MENU -> the line-emission retry count.
    #
    # These are RUN-LOCAL variables on purpose. `cfg` is a shared dataclass whose
    # fingerprint is already stamped into the certificate; mutating it would make
    # the fingerprint a lie and would leak one task's allocation into the next.
    # The allocation itself is recorded in the cert so an ablation can read it.
    #
    # FAIL-OPEN: any error anywhere leaves both values at their cfg defaults.
    #
    # RAISE-ONLY, and this is the load-bearing decision. The allocator's own
    # floors (BALLOT=1, MENU=1) are floors on ITS policy, not on this harness:
    # a naive `depth = per_stage[BALLOT]` cut the line-ballot budget from the
    # measured 8 to 1, which makes `len(probe_lines) >= 2` false and DISABLES the
    # ballot outright — the single highest-value mechanism in the ladder autopsy,
    # switched off by the feature meant to fund it. The knob values in gym_config
    # are measured settings; the allocator exists to buy MORE depth where the
    # task is uncertain, never to spend a measured default. So the applied value
    # is max(cfg default, allocator floor, allocated depth) and the cert records
    # all three.
    ballot_max_calls = int(cfg.fix.line_ballot_max_calls)
    fix_retries = int(cfg.fix.retries)
    if getattr(cfg.attention, "enabled", False):
        try:
            from detreason import attention as _attn
            _pol = _attn.AllocatorPolicy(
                reserve_fraction=float(cfg.attention.reserve_fraction),
                hysteresis_ratio=float(cfg.attention.hysteresis_ratio),
                consecutive_required=int(cfg.attention.hysteresis_checkpoints))
            _state = _attn.TaskState(budget=float(cfg.attention.budget),
                                     stakes_tier=str(cfg.attention.stakes_tier),
                                     task_key=task_text[:80])
            _alloc = _attn.allocate(_state, _pol)
            _per = dict(_alloc.per_stage)
            _floors = {s: _pol.stages[s].floor for s in _attn.STAGES}
            _ballot = max(ballot_max_calls, int(_floors[_attn.BALLOT]),
                          int(_per.get(_attn.BALLOT, 0)))
            _menu = max(fix_retries, int(_floors[_attn.MENU]),
                        int(_per.get(_attn.MENU, 0)))
            # RAISE-ONLY again, and capped by the task-wide request ceiling: the
            # allocator may buy an uncertain task a second look at the evidence,
            # it may never buy it more than the task budget it is spending from.
            _req = min(int(cfg.fix.info_request_budget_per_task),
                       max(int(getattr(cfg.fix, "info_request_budget_per_menu", 1)),
                           int(_per.get(_attn.MENU, 0))))
            fx.request_budget_per_menu = _req
            cert["attention"] = {
                "per_stage": _per,
                "applied": {"line_ballot_max_calls": _ballot,
                            "fix_retries": _menu,
                            "info_request_budget_per_menu": _req},
                "cfg_defaults": {"line_ballot_max_calls": ballot_max_calls,
                                 "fix_retries": fix_retries},
                "floors": _floors,
                "stakes_tier": str(cfg.attention.stakes_tier),
                "budget": float(cfg.attention.budget)}
            ballot_max_calls, fix_retries = _ballot, _menu
        except Exception as _e:
            cert.setdefault("warnings", []).append(
                f"attention allocator unavailable ({type(_e).__name__})")
    root, files = load_repo(root, stats=io_stats)
    # Validate an oracle-pinned target before any model-backed classification or
    # retrieval work. A missing target is a repository-resolution miss, not a
    # reason to spend a class/menu call that cannot possibly repair it.
    if target_file is not None and target_file not in files:
        _record_retrieval_miss(
            cert,
            "repo_root",
            target_file,
            (),
            "pinned target file is absent from the resolved repository root",
        )
        cert["corpus_files"] = len(files)
        cert["reason"] = (
            "locate: pinned target file %r is absent from repo" % target_file
        )
        return cert
    idx, df, n, adj = build_index(files)
    cert["corpus_files"] = len(files)
    # brace-sanitize the task prose (standing rule: brace literals hang decisions)
    loc_task = re.sub(r"\{[^}]*\}", "the expected value", task_text) \
        if cfg.facts.brace_sanitize else task_text
    # EXACT-SYMBOL ANCHORS (g1 stage 2; smoke-1 lesson: the ticket NAMED the
    # function and the cascade ignored it): code-like tokens from the raw task
    # (snake_case identifiers, ClassNames, .attrs) exact-matched against defs.
    sym_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task_text))
    code_like = {t for t in sym_tokens if "_" in t or (t[:1].isupper() and len(t) > 3)}
    attr_mentions = set(re.findall(r"\.([a-z_][a-z0-9_]{2,})", task_text))
    fn_anchors, cls_anchors = [], []
    if cfg.locate.anchors:
        for path, src in files.items():
            for tok in code_like:
                if re.search(rf"\bdef {re.escape(tok)}\b", src):
                    fn_anchors.append((path, tok))
                elif re.search(rf"\bclass {re.escape(tok)}\b", src):
                    cls_anchors.append((path, tok))
    # a task-named FUNCTION is where the bug lives; a named class is usually the
    # referenced contract (smoke-2: class file outranked the function file)
    anchors = fn_anchors + cls_anchors
    cert["anchors"] = anchors[:8]
    # ---- FACTS channel (knob-routed; injected into every downstream prompt) ----
    facts = []
    if cfg.facts.inject_class_attrs:
        for _, tok in cls_anchors[:3]:
            for p2, s2 in files.items():
                if re.search(rf"\bclass {re.escape(tok)}\b", s2):
                    attrs = sorted({a for a in re.findall(r"^\s+(\w+)\s*[:=]", s2,
                                                          flags=re.M)
                                    if not (cfg.fix.exclude_const_attrs and a.isupper())
                                    and not a.startswith("_")})[:12]
                    if attrs:
                        facts.append(f"class {tok} attributes: {', '.join(attrs)}")
                    break
    if cfg.facts.inject_task_named_files:
        named = [p for p in files
                 if any(t.lower() in pathlib.Path(p).stem.lower() for t in code_like)]
        if named:
            facts.append("files named by the task: " + ", ".join(sorted(named)[:5]))
    base_task = ("FACTS:\n" + "\n".join(f"  - {f}" for f in facts) + "\n\n" + loc_task) \
        if facts else loc_task
    cert["facts"] = facts
    # Class inference: C0 taxonomy FIRST (deterministic-before-model — the
    # refactor regressed this to model-only and the unified gym showed it),
    # model menu only for the residue.
    _c0_cls = infer_symptom_class(task_text)
    cls = _c0_cls
    class_parse_by = "c0" if _c0_cls else None
    if cls is None:
        class_prompt = CLASS_Q.format(task=base_task)
        text, _ = _call(
            class_prompt, seed=cfg.seed, url=cfg.url,
            max_tokens=budget_for_stage(cfg, "menu", class_prompt),
            enable_thinking=stage_thinking(cfg, "menu"), label="class",
            continuation=CONTINUATION_Q)
        m = re.search(r'"pick"\s*:\s*"([A-J])"', text or "")
        if m:
            letter = m.group(1)
            class_parse_by = "strict"
        else:
            letter = _lenient_letter(text, list("ABCDEFGHIJ"))
            if letter:
                class_parse_by = "lenient"
            else:
                letter = _prose_letter(text, list("ABCDEFGHIJ"))
                class_parse_by = "prose" if letter else None
        cls = CLASS_MAP.get(letter) if letter else None
    cert["stages"].append({"class": cls or "unknown",
                           "class_by": "c0" if _c0_cls else "menu",
                           "class_parse_by": class_parse_by,
                           "complex_class": cls in COMPLEX_CLASSES})
    # PLAN-PINNED FILE: the caller already knows the target file (chain edit).
    # Skipping locate entirely prevents the fixer re-locating to a different
    # file than the plan declared. Strictly additive: None -> unchanged path.
    # shared across both locate paths so downstream stages never NameError when
    # the caller pinned a file (or func_index was unavailable).
    fidx = None
    fxi = None
    franked = None          # rank_functions output, kept for the pool-recall arm
    sym_anchor = None
    over_cand = []          # recall-arm overflow: a SECOND menu, never a wider one
    over_disp = []          # its evidence cards, in lockstep with over_cand
    anchor_files = []       # hard anchors protected by the locate ranker
    recall_paths = []       # append-only construct recall, never rank-promoted
    _file_rank_precut = False
    _file_rctx = None       # info-request context; set by the file-menu branch
    _pick_file = None       # selector-aware file picker; built by that branch

    # RANKING QUERY: the FACTS block (class attribute lists) is written for the
    # DECIDING model, not for BM25 — a dozen injected attribute words are extra
    # query terms that rerank the corpus toward whatever else mentions them.
    # The knob decides whether ranking sees them; every PROMPT still does.
    rank_task = base_task if cfg.locate.rank_query_facts else loc_task
    # DUNDER EXPANSION (rung-4000 autopsy, ranked fix #6b). RETRIEVAL ONLY — the
    # expansion never reaches a prompt, so no menu question is reworded by it.
    if getattr(cfg.locate, "dunder_expansion", False):
        _expanded = expand_dunder_query(rank_task)
        if _expanded != rank_task:
            cert["dunder_expansion"] = sorted(
                set(_expanded.split()) - set(rank_task.split()))[:12]
            rank_task = _expanded
    model_menu_k = min(cfg.locate.menu_k, MAX_MODEL_FILE_MENU_OPTIONS)
    if target_file is not None:
        cand = [target_file]
        locate_baseline = list(cand)
        rerank_mode = _rerank_compose_mode(cfg)
        rerank_budget = _rerank_compose_budget(cfg)
        rank_proposal = list(cand)
        locate_rank_rec = None
        log, func_evidence = {}, {}
        fpick = target_file
        cert["stages"].append({"file_by": "plan_pinned"})
    else:
        # locate cascade — anchor files ranked FIRST (deterministic beats retrieval)
        log = {}
        func_evidence = {}
        sym_anchor = None      # (path, function) from a UNIQUE symptom-phrase match
        if cfg.locate.function_granularity:
            # FUNCTION-GRANULARITY view (a_mechanical: canonical unit = function;
            # +4 menu recall offline). Menu order from the FUSED lexical-led rank
            # (cycle-2/3 lesson: construct-union feeds the POOL, never menu order).
            try:
                import func_index as fxi
                fidx = fxi.build_func_index(files)
                fpool, ranked = fxi.function_pool(rank_task, cls, fidx,
                                                  pool_cap=cfg.locate.pool_cap)
                # APPEND-ONLY RECALL ARM: the construct-union pool was computed
                # and then DISCARDED — the menu came from the fused rank alone,
                # so the construct-only arm (the file view's whole recall edge on
                # vocabulary-disjoint symptoms) reached nothing. Its paths are now
                # APPENDED after the ranked menu; existing entries never move.
                # Measured 08-28: gym menu recall 17/21 -> 20/21.
                franked = ranked
                pool_paths = list(dict.fromkeys(r.path for r in fpool))
                model_menu_k = min(cfg.locate.menu_k, MAX_MODEL_FILE_MENU_OPTIONS)
                fmenu, membership = fxi.file_menu_from_functions(
                    ranked, k=_file_rank_window(cfg, model_menu_k),
                    append_paths=pool_paths,
                    append_k=cfg.locate.recall_append_k)
                # Keep recall-only additions out of the pre-cut ranking window.
                # They remain an overflow arm, exactly as before.
                recall_paths = [p for p, ev in fmenu
                                if isinstance(ev, dict)
                                and ev.get("source") == "recall_append"]
                cand = [p for p, ev in fmenu
                        if not (isinstance(ev, dict)
                                and ev.get("source") == "recall_append")]
                func_evidence = {p: ev for p, ev in fmenu}
                log["func_membership_top"] = [
                    m for m in membership[:40] if isinstance(m, dict)]
                # SYMPTOM ANCHOR: a quoted phrase in the ticket whose words are
                # exactly a function's name ('map level' -> _map_level). Same
                # evidence class as the snake_case anchor above, for phrases the
                # ticket spelled with spaces. Only a UNIQUE match pins.
                if cfg.locate.symptom_anchors:
                    sa = fxi.name_phrase_anchors(task_text, fidx)
                    uniq = {(r.path, r.qualname.split(".")[-1]) for r in sa}
                    cert["symptom_anchors"] = sorted(uniq)[:8]
                    if len(uniq) == 1:
                        sym_anchor = next(iter(uniq))
            except Exception as e:
                cert.setdefault("warnings", []).append(
                    f"func_index unavailable ({type(e).__name__}); file view")
                cand = fx.cascade(rank_task, idx, df, n, adj, log,
                                  cls=cls, all_files=files)
        else:
            cand = fx.cascade(rank_task, idx, df, n, adj, log, cls=cls, all_files=files)
        anchor_files = list(dict.fromkeys(p for p, _ in anchors))
        cand = anchor_files[:4] + [p for p in cand if p not in anchor_files]
        # RANK BEFORE CUT. The old path passed only the first eight files to
        # rank_file_menu, so a gold file at rank 9+ was impossible to recover
        # even when it was already in the candidate pool. This window is bounded
        # and the model still receives only `model_menu_k` primary options.
        model_menu_k = min(cfg.locate.menu_k, MAX_MODEL_FILE_MENU_OPTIONS)
        _rank_before = list(cand)
        _rank_after = list(cand)
        _rank_fan = None
        if (getattr(cfg.locate, "file_rank_fusion", False)
                and len(cand) > 1 and len(set(cand)) == len(cand)):
            if getattr(cfg.locate, "fan_in_penalty", False):
                try:
                    _rank_fan = import_fan_in(files)
                except Exception as _e:      # law 1: fail OPEN to no penalty
                    cert.setdefault("warnings", []).append(
                        f"fan-in unavailable ({type(_e).__name__})")
            _definition_guard_active = bool(
                getattr(cfg.locate, "file_rank_definition_guard", False)
                and len(cand) > model_menu_k)
            _rank_incumbent = None
            if _definition_guard_active:
                _rank_incumbent = rank_file_menu(
                    cand[:model_menu_k], func_evidence, idx, task_text,
                    [p for p, _ in fn_anchors], cfg, fan_in=_rank_fan,
                    files=files, compose=False)
            _rank_after = rank_file_menu(
                cand, func_evidence, idx, task_text,
                [p for p, _ in fn_anchors], cfg, fan_in=_rank_fan,
                files=files, baseline=_rank_incumbent,
                compose=not _definition_guard_active)
            if sorted(_rank_after) == sorted(_rank_before):
                cand = list(_rank_after)
                _file_rank_precut = True
                _primary_before = _rank_before[:model_menu_k]
                _primary_after = _rank_after[:model_menu_k]
                _fan_k = int(getattr(cfg.locate, "fan_in_k", 0) or 0)
                _fan_m = float(getattr(cfg.locate, "fan_in_m", 1.0) or 1.0)
                _fan_menu = {p: (_rank_fan or {}).get(p, 0)
                             for p in _primary_after}
                _fan_penalised = sorted(
                    p for p, count in _fan_menu.items() if count > _fan_k)
                cert["fan_in"] = {"k": _fan_k, "m": _fan_m,
                                   "penalised": _fan_penalised,
                                   "menu": _fan_menu}
                cert["file_rank_fusion"] = {
                    "reordered": _primary_after != _primary_before,
                    "before": _primary_before,
                    "after": list(_primary_after),
                    "proposed_after": list(_primary_after),
                    "compose": _rerank_compose_mode(cfg),
                    "budget": _rerank_compose_budget(cfg),
                    "applied": True,
                    "window": len(_rank_before),
                }
                cert["stages"].append({"locate_rank": {
                    "fan_in_penalty": bool(getattr(
                        cfg.locate, "fan_in_penalty", False)),
                    "fan_in_k": _fan_k, "fan_in_m": _fan_m,
                    "fan_in_available": _rank_fan is not None,
                    "penalised": _fan_penalised, "fan_in_menu": _fan_menu,
                    "reordered": _primary_after != _primary_before,
                    "before": _primary_before, "after": _primary_after,
                    "proposed_after": _primary_after,
                    "compose": _rerank_compose_mode(cfg),
                    "budget": _rerank_compose_budget(cfg),
                    "applied": True, "window": len(_rank_before),
                }})
        # Recall candidates stay useful for ranking/ledger evidence, but the
        # model-facing menu stays within its calibrated eight-choice contract.
        # The cap decides where the MODEL menu ends, not where the candidate list
        # ends: everything past it becomes the overflow menu below. Truncating
        # here instead would silently throw the recall arm away (measured: 21/21
        # gold files reachable with the split, 16/21 without).
        cand = cand[:model_menu_k]
        for _p in recall_paths:
            if len(cand) >= model_menu_k + max(0, cfg.locate.recall_append_k):
                break
            if _p not in cand:
                cand.append(_p)
        # a unique symptom-phrase anchor is a candidate too, even if retrieval
        # never surfaced its file (append-only: it goes at the END).
        if sym_anchor and sym_anchor[0] not in cand:
            cand.append(sym_anchor[0])
        # AMBIGUOUS-ANCHOR RECALL ARM (E52) — append-only, overflow-only.
        # A quoted phrase matching MANY function names cannot pin (uniqueness is
        # what makes a pin safe), and the set was then thrown away entirely. It is
        # still exact-name evidence, so its top-K paths by the fused score join
        # the candidate list at the END. Everything above is untouched, so the
        # primary menu (cand[:model_menu_k]) keeps exactly the members, order and
        # evidence cards it had — these paths can only reach the overflow menu,
        # which is asked ONLY when the primary produced no pick.
        # Measured 2026-08-28: rescues the sole gym
        # recall miss, models/learner_concept_gap.py::__repr__, whose gold file
        # is rank 40 of the fused ranking and rank 0 of its own anchor set.
        anchor_k = max(0, int(getattr(cfg.locate, "anchor_recall_k", 0) or 0))
        if (cfg.locate.symptom_anchors and anchor_k and sym_anchor is None
                and fidx is not None):
            try:
                arecs = fxi.rank_name_phrase_anchors(task_text, cls, fidx)
            except Exception as _e:            # recall is additive, never fatal
                arecs = []
                cert.setdefault("warnings", []).append(
                    f"anchor recall unavailable ({type(_e).__name__})")
            appended = []
            for r in arecs:
                if len(appended) >= anchor_k:
                    break
                if r.path in cand or r.path in appended:
                    continue
                appended.append(r.path)
            # width budget: the overflow menu is a menu too (#menu-width lesson),
            # so it never grows past the calibrated eight-choice contract.
            room = max(0, (min(cfg.locate.menu_k, MAX_MODEL_FILE_MENU_OPTIONS)
                           + MAX_MODEL_FILE_MENU_OPTIONS) - len(cand))
            appended = appended[:room]
            cand.extend(appended)
            if appended:
                cert["anchor_recall_appended"] = appended
        # POOL-RECALL WIDENING (wired 2026-08-29)
        #
        #   BM25 top-W  ->  keep candidates with >= 2 distinct symptom-token
        #   hits  ->  rerank  ->  cap 64  ->  APPEND (never insert).
        #
        # The four known pool misses at MC scale sit at Stage-A ranks 19-58,
        # i.e. inside a W=100 sweep and outside today's cap. The >=2-hit gate is
        # the precision guard: the doc proves the gold cannot be removed by it
        # (every gold in the slice carries at least three symptom tokens) while
        # a lone shared token can no longer drag a file in.
        #
        # WHY APPEND-ONLY IS THE WHOLE SAFETY ARGUMENT: retrieval alone cannot
        # guarantee zero false locates -- a wrong file may legitimately carry
        # two symptom tokens. The guarantee comes from ROLLOUT: the primary
        # menu prefix keeps exactly the options, order and evidence cards it
        # had, so these paths can only ever reach the OVERFLOW menu, which is
        # asked only when the primary produced no pick. Delta false-locate = 0
        # by construction, and the widening still buys gold-in-pool.
        _pr_rec = {"stage_id": "pool_recall", "status": "DISABLED",
                   "pool_size": 0, "baseline_size": len(cand), "added": [],
                   "rejected_count": 0, "truncated": False, "stage_a_size": 0,
                   "min_hits": 0, "pool_cap": 0,
                   "input_digest": "", "output_digest": ""}
        if getattr(cfg.locate, "pool_recall", False) and franked:
            try:
                import pool_recall as _prc
                _qtok = _pool_recall_query_tokens(task_text)
                _recs = [r for _s, r, _e in franked if not r.is_module]
                _by_id = {r.func_id: r for r in _recs}
                _score = {r.func_id: s for s, r, _e in franked}
                _res = _prc.two_stage_pool(
                    _qtok, [r.func_id for r in _recs],
                    score_fn=lambda c: _score.get(c, 0.0),
                    hits_fn=lambda c: {t for t in _qtok
                                       if t in (_by_id[c].bag if c in _by_id
                                                else {})},
                    W=int(getattr(cfg.locate, "pool_recall_w", 100)),
                    min_hits=int(getattr(cfg.locate, "pool_recall_min_hits", 2)),
                    pool_cap=int(getattr(cfg.locate, "pool_recall_cap", 64)),
                    anchors=tuple(r.func_id for r in _recs
                                  if r.path in anchor_files))
                _pr_rec = _res.record()
                _pr_rec["status"] = "applied"
                _pr_rec["baseline_size"] = len(cand)
                # func_ids -> paths, dedup, then the SAME width budget the
                # anchor arm obeys: the overflow menu is a menu too.
                _wide = []
                for _cid in _res.pool:
                    _p = _by_id[_cid].path if _cid in _by_id else None
                    if _p and _p not in cand and _p not in _wide:
                        _wide.append(_p)
                # OVERFLOW-ONLY, ENFORCED. "Append-only" is not by itself the
                # zero-false-locate guarantee: the primary menu is `cand`'s
                # first `model_menu_k` entries, so if the cascade returned
                # FEWER than that, an appended path would land INSIDE the
                # primary menu and the guarantee would silently evaporate on
                # exactly the thin-candidate tasks where a wrong pick is most
                # likely. So the arm refuses to fill a short primary menu.
                _primary_w = min(cfg.locate.menu_k, MAX_MODEL_FILE_MENU_OPTIONS)
                _room = max(0, (_primary_w + MAX_MODEL_FILE_MENU_OPTIONS)
                            - len(cand))
                if len(cand) < _primary_w:
                    _wide = []
                    _pr_rec["status"] = "applied"
                    _pr_rec["reason"] = ("primary menu is short (%d < %d); "
                                         "appending would widen it"
                                         % (len(cand), _primary_w))
                _wide = _wide[:_room]
                cand = list(_prc.append_only_union(cand, _wide))
                _pr_rec["added"] = list(_wide)
            except Exception as _e:          # law 1: recall is additive, never fatal
                _pr_rec = dict(_pr_rec, status="unavailable",
                               reason="%s: %s" % (type(_e).__name__, _e))
                cert.setdefault("warnings", []).append(
                    f"pool recall unavailable ({type(_e).__name__})")
        cert["stages"].append({"pool_recall": _pr_rec})
        # C0: exactly one uniquely-named function anchor -> its file, no menu
        fn_files = list(dict.fromkeys(p for p, _ in fn_anchors))
        forced_file = fn_files[0] if (len(fn_files) == 1 and
                                      cfg.locate.anchor_forces_file) else None
        if not cand:
            cert["reason"] = "locate: no candidates survived the cascade"
            return cert
        q = words(base_task)
        if cfg.locate.evidence_cards:
            displays = []
            # MENU-WIDE document frequency, computed BEFORE any card is rendered:
            # "distinctive first" is a statement about this menu, so it cannot be
            # decided one option at a time (the 'and, for, the, that' card).
            _raw_matched = []
            for p in cand:
                _ev = func_evidence.get(p)
                _lex = sorted(q & set(idx.get(p) or {}),
                              key=lambda w: -idx[p][w])
                _raw_matched.append(card_matched_words(
                    _ev if isinstance(_ev, dict) else None, _lex))
            card_df = menu_doc_frequency(_raw_matched)
            for p in cand:
                ev = func_evidence.get(p)
                # function-level evidence card: WHICH function and WHY it ranked
                # (the key is `best_func` — reading `name` here made every card
                # say None and killed the promotion dividend below; #223).
                # Rendered through evidence_card so a `<module>` pseudo-record
                # and the raw ranking floats never reach the model (autopsy #5).
                lex_matched = sorted(q & set(idx.get(p) or {}),
                                     key=lambda w: -idx[p][w])
                displays.append(evidence_card(p, ev if isinstance(ev, dict) else None,
                                              lex_matched, cls=cls,
                                              src=files.get(p), df=card_df))
            # two options that read the same have no correct answer
            displays = distinguish_cards(cand, displays)
            file_q = FILE_MENU_EV_Q
        else:
            displays = list(cand)
            file_q = FILE_MENU_Q
        # MENU WIDTH IS A HARD BUDGET, measured 2026-08-28 on the live gym: the
        # 1B answered 39/40 file menus at n=8 and only 10/17 at n=12 — a wider
        # menu buys recall and pays for it in no-picks. So the recall arm does
        # NOT widen the menu: the primary menu keeps exactly the options and the
        # order it always had, and the extra candidates form an OVERFLOW menu
        # that is asked ONLY when the primary produced no pick.
        primary_n = model_menu_k
        over_cand, over_disp = cand[primary_n:], displays[primary_n:]
        cand, displays = cand[:primary_n], displays[:primary_n]
        locate_baseline = list(cand)
        rerank_mode = _rerank_compose_mode(cfg)
        rerank_budget = _rerank_compose_budget(cfg)
        rank_proposal = list(locate_baseline)
        locate_rank_rec = None
        # RANK FUSION (E51) — deliberately AFTER the primary/overflow split, so
        # it permutes the model-facing window and can never move the split
        # boundary. Menu membership, width and recall are identical with the
        # knob on or off; only position moves. displays travel in lockstep so
        # every option keeps its own evidence card.
        if (getattr(cfg.locate, "file_rank_fusion", False)
                and not _file_rank_precut and len(cand) > 1
                and len(set(cand)) == len(cand)):
            pre = list(cand)
            # the RAW ticket, not base_task: the FACTS block is written for the
            # deciding model and its injected words would rerank the menu, which
            # is the rank_query_facts defect (E42) by another door.
            # FAN-IN, computed ONCE per task and only when the knob is on: the
            # AST pass is over the whole corpus, so paying for it on every run
            # to feed a dormant experiment would be a real cost for no signal.
            _fan = None
            if getattr(cfg.locate, "fan_in_penalty", False):
                try:
                    _fan = import_fan_in(files)
                except Exception as _e:      # law 1: fail OPEN to no penalty
                    cert.setdefault("warnings", []).append(
                        f"fan-in unavailable ({type(_e).__name__})")
                else:
                    cert["fan_in"] = {
                        "k": int(getattr(cfg.locate, "fan_in_k", 0) or 0),
                        "m": float(getattr(cfg.locate, "fan_in_m", 1.0) or 1.0),
                        "penalised": sorted(
                            p for p in cand
                            if _fan.get(p, 0) > int(
                                getattr(cfg.locate, "fan_in_k", 0) or 0)),
                        "menu": {p: _fan.get(p, 0) for p in cand}}
            rank_proposal = rank_file_menu(
                cand, func_evidence, idx, task_text,
                [p for p, _ in fn_anchors], cfg, fan_in=_fan,
                baseline=pre, compose=False)
            if sorted(rank_proposal) == sorted(pre):
                cert["file_rank_fusion"] = {
                    "reordered": rank_proposal != pre,
                    "before": pre, "after": list(pre),
                    "proposed_after": list(rank_proposal),
                    "compose": rerank_mode, "budget": rerank_budget,
                    "applied": rerank_mode == "joint"}
            # THE STAGE RECORD, added 2026-08-29. The two top-level cert keys
            # above are NOT in the gym ledger: `probes/unified_gym.py:run_one`
            # copies only verdict / margins / stages / usage_totals off the
            # certificate, so `cert["fan_in"]` and `cert["file_rank_fusion"]`
            # were written and then dropped on the floor. The consequence was a
            # question nobody could answer from the record — "did the fan-in
            # penalty apply on this run?" — while the knob read as ON in
            # `config`. A `stages` entry survives the row writer, so the
            # ANSWER now travels with the run. Same facts, no second authority:
            # this is a projection of the two dicts written directly above.
            locate_rank_rec = {
                "fan_in_penalty": bool(getattr(cfg.locate, "fan_in_penalty", False)),
                "fan_in_k": int(getattr(cfg.locate, "fan_in_k", 0) or 0),
                "fan_in_m": float(getattr(cfg.locate, "fan_in_m", 1.0) or 1.0),
                "fan_in_available": _fan is not None,
                "penalised": list((cert.get("fan_in") or {}).get("penalised") or []),
                "fan_in_menu": dict((cert.get("fan_in") or {}).get("menu") or {}),
                "reordered": False,
                "before": list(pre), "after": list(pre),
                "proposed_after": list(rank_proposal),
                "compose": rerank_mode, "budget": rerank_budget,
                "applied": rerank_mode == "joint",
            }
            cert["stages"].append({"locate_rank": locate_rank_rec})
        # CONSULTATION CHANNELS (blame + case learning), deliberately AFTER the
        # primary/overflow split and after rank fusion: like fusion they may
        # only PERMUTE the model-facing window, so menu membership, width and
        # recall are identical with the knobs on or off. displays travel in
        # lockstep so every option keeps its own evidence card.
        if (len(cand) > 1 and len(set(cand)) == len(cand)
                and (getattr(cfg.locate, "blame_channel", False)
                     or getattr(cfg.locate, "case_learning", False))):
            _content_scores = {}
            for _p in locate_baseline:
                _ev = func_evidence.get(_p)
                if isinstance(_ev, dict):
                    _content_scores[_p] = _ev.get("adj_score")
            _post_c, _consult_delta = _locate_consultations(
                locate_baseline, cfg=cfg, cert=cert, files=files,
                task_key=task_text[:64], repo_root=root, anchor_files=anchor_files,
                baseline=locate_baseline, max_shift=rerank_budget,
                content_scores=_content_scores, apply=False, return_delta=True)
        else:
            _consult_delta = {}
            # A DISABLED record is not noise: it is what distinguishes "the
            # knob is off" from "somebody forgot to instrument this stage".
            for _ch in ("blame", "case_learning"):
                cert["stages"].append({"locate_consultation":
                                       _blank_consult_record(_ch)})
        # One composition point for the entire locate stack. Each proposal is
        # measured from `locate_baseline`, so sequential adjusters cannot spend
        # the same displacement budget repeatedly.
        _orders = (rank_proposal,) if rank_proposal != locate_baseline else ()
        cand = _joint_rerank(
            locate_baseline, proposed_orders=_orders,
            deltas=(_consult_delta,), pinned=anchor_files,
            mode=rerank_mode, max_shift=rerank_budget)
        if sorted(cand) == sorted(locate_baseline):
            _disp_by = dict(zip(locate_baseline, displays))
            displays = [_disp_by[p] for p in cand]
        if locate_rank_rec is not None:
            locate_rank_rec["reordered"] = cand != locate_baseline
            locate_rank_rec["after"] = list(cand)
            if cert.get("file_rank_fusion"):
                cert["file_rank_fusion"]["reordered"] = cand != locate_baseline
                cert["file_rank_fusion"]["after"] = list(cand)
        # instrumentation: recall-vs-pick is only separable if the trace records
        # what was reachable but never shown (the diagnosis engine's ask).
        cert["overflow_candidates"] = list(over_cand)
        cert.setdefault("retrieval", {})["file_menu"] = (
            list(cand) + list(over_cand)
        )
        # INFO-REQUEST context for the file menu: an option is a PATH, and the
        # function that made it rank is the one the model would want to read
        # (func_evidence.best_func — the same record the evidence card was
        # rendered from, so a request shows the code behind the card). Built
        # unconditionally now: it is pure construction, no call and no prompt, and
        # the insufficient-menu escape below needs it on paths where the primary
        # menu was never asked.
        def _file_locate(opt, _fe=func_evidence):
            _ev = _fe.get(opt)
            _bf = (_ev.get("best_func") if isinstance(_ev, dict) else None)
            return {"path": opt,
                    "func": (_bf.split(".")[-1]
                             if _bf and _bf != "<module>" else None)}

        _file_rctx = RequestContext("file", files, _file_locate, cfg)
        # ---- SELECTOR-AWARE FILE PICK (E100 port of the binary selector) -----
        # The file stage A-LOCKED exactly like the mutation menu did: E100 caught
        # it answering index 0 twice with gold at index 3, and the same shape is
        # in every results-ledger row since 1788010000. The cure is the one that
        # was already proven at the mutation menu — ask ONE independent yes/no
        # question per candidate, so there is no menu order for the answer to
        # track, let CODE rank the answers, and spend one INVERTED confirm that
        # can only veto.
        #
        # WHAT A PROBE SEES (`probe_snippet`): the candidate's path plus the top
        # matching lines that put it on the menu. Deliberately NOT whole files —
        # at N probes that is unaffordable, and it would turn a relevance question
        # into a recall one.
        #
        # THE INFO-REQUEST WRAPPER IS LETTER-PATH ONLY, for the same structural
        # reason it is at the mutation menu: `REQUEST(SHOW_SOURCE, B)` needs a
        # lettered option list to point at, and a binary probe sees exactly one
        # candidate with nothing to name. `_file_rctx` is still built above and
        # still travels the letter path untouched.
        _file_selector = str(getattr(cfg.fix, "file_selector", "letter")
                             or "letter")

        def _file_snip(_p, _q=None):
            _ev = func_evidence.get(_p)
            _bf = _ev.get("best_func") if isinstance(_ev, dict) else None
            return probe_snippet(files.get(_p, ""), q,
                                 focus=(_bf if _bf and _bf != "<module>" else None))

        def _pick_file(_opts, _disps, _label):
            """One file menu, by whichever selector is armed."""
            if _file_selector == "pairwise" and len(_opts) > 1:
                # RANK-1 FRAMING. The card each side shows is the same
                # evidence card the letter menu renders plus the matching lines
                # the absolute probe used — symmetric by construction, because a
                # comparison between differently-shaped cards measures the shapes.
                def _pair_q(pa, da, pb, db):
                    return FILE_PAIR_Q.format(
                        task=base_task, path_a=pa, path_b=pb,
                        card_a="%s\n%s" % (da, _file_snip(pa)),
                        card_b="%s\n%s" % (db, _file_snip(pb)))

                return fx.menu_pick_pairwise(list(_opts), list(_disps), _label,
                                             _pair_q)
            if _file_selector == "binary" and _opts:
                _snips = {p: _file_snip(p) for p in _opts}
                # CONFIRM IS OFF BY DEFAULT AT THIS STAGE (measured 8/8 vetoes,
                # 0/8 confirms — see gym_config.location_probe_confirm). Passing
                # None is how `menu_pick_binary` is told not to spend the call:
                # the ranked winner then stands on its probe evidence alone.
                _cq = None
                if getattr(cfg.fix, "location_probe_confirm", False):
                    def _cq(o):
                        return FILE_CONFIRM_Q.format(task=base_task, path=o,
                                                     snippet=_snips[o])
                return fx.menu_pick_binary(
                    list(_opts), list(_disps), _label,
                    lambda o: FILE_PROBE_Q.format(task=base_task, path=o,
                                                  snippet=_snips[o]),
                    _cq,
                    probe_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"in_file": "<yes|no>"}.'),
                    confirm_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"unrelated": "<yes|no>"}.'))
            return fx.menu_pick(
                lambda menu: file_q.format(task=base_task, menu=menu),
                _opts, _disps, _label, rctx=_file_rctx)

        def _file_selector_stage(_label):
            """Per-candidate probe scores into the CERTIFICATE, not only into
            `margins` — the calibration pass reads the cert, and a selector whose
            scores are unrecorded cannot be second-guessed."""
            _r = fx.margins[-1] if fx.margins else {}
            if not (isinstance(_r, dict) and _r.get("menu") == _label):
                return
            if _file_selector == "binary" and _r.get("binary_probe"):
                cert["stages"].append({"file_selector": _r["binary_probe"]})
            elif _file_selector == "pairwise" and _r.get("pairwise"):
                cert["stages"].append({"file_selector": _r["pairwise"]})

        if sym_anchor is not None and cfg.locate.anchor_forces_file:
            fpick = sym_anchor[0]
            cert["stages"].append({"file_by": "symptom_anchor"})
        elif forced_file is not None:
            fpick = forced_file
            cert["stages"].append({"file_by": "unique_fn_anchor"})
        else:
            fpick = _pick_file(cand, displays, "file")
            _file_selector_stage("file")
            if fpick is None and over_cand:
                # second chance, never a wider first chance
                fpick = _pick_file(over_cand, over_disp, "file_overflow")
                _file_selector_stage("file_overflow")
                if fpick is not None:
                    cert["stages"].append({"file_by": "recall_overflow"})
    # ---- INSUFFICIENT-MENU ESCAPE (2026-08-29 final-four autopsy, task #4) ----
    # The recall arm was DEAD BY WIRING. `over_cand` was consulted on exactly one
    # condition — `fpick is None` — so it could only ever help a model that had
    # already ABSTAINED. MEASURED on `models/learner_concept_gap.py::__repr__`:
    # the model answered the primary menu unanimously and confidently
    # (`utils/log_context.py`, 3/3 votes, margin 0.92), the gold file was in
    # `over_cand`, the overflow was never asked, the picked function produced
    # `mutations: 0`, line emission found nothing admissible, and the run
    # ABSTAINed. A confident wrong pick is precisely the case the recall arm
    # exists for, and it was the one case it could not reach.
    #
    # The escape is CODE, not a second opinion: a `none_flip` cannot live in a
    # function holding no `is`/`is not` comparison, so a picked file with zero
    # constructs of the symptom's own class cannot host the bug that was
    # reported. When that is true AND an overflow candidate demonstrably CAN
    # host it, the menu we asked was the wrong menu and we re-ask on the
    # feasible ones. Two properties keep this from being a licence to re-roll:
    #   * it fires only on a provable dead end, so it can only convert runs that
    #     were already scoring zero — never a run that was about to succeed;
    #   * the re-ask is restricted to FEASIBLE overflow files, so it is a
    #     narrower question than the one that was already answered, never a
    #     wider one (the #menu-width lesson).
    #
    # SCOPED TO A GUESS. The escape may only second-guess a MENU pick. When the
    # file came from a unique symptom anchor or a unique function anchor, the
    # ticket NAMED that function and the pin is proven evidence, not a choice
    # among options — overriding it with a re-ask would trade a proof for a
    # guess, which is the wrong direction and the opposite of what this escape
    # is for. Those two branches set `file_by` before this point, so a pinned
    # run carries a `file_by` stage and is left alone.
    # `plan_pinned` belongs here for the same reason the other two do: the caller
    # NAMED the file, so there is no menu pick to second-guess and no evidence to
    # demand of a choice that was never made.
    _pinned = any(s.get("file_by") in ("symptom_anchor", "unique_fn_anchor",
                                       "plan_pinned")
                  for s in cert["stages"] if isinstance(s, dict))
    if fpick is not None and cls and over_cand and not _pinned:
        try:
            _feas = {p: (fxi._count_constructs(files.get(p, "")) or {}).get(cls, 0)
                     for p in [fpick] + list(over_cand)}
        except Exception as _e:
            _feas = {}
            cert.setdefault("warnings", []).append(
                f"insufficient-menu escape unavailable ({type(_e).__name__})")
        _alt = [p for p in over_cand if _feas.get(p, 0) > 0]
        if _feas.get(fpick, 1) == 0 and _alt and _pick_file is not None:
            _disp_by = dict(zip(over_cand, over_disp))
            # THE SAME SELECTOR as the menu it is second-guessing. Re-asking a
            # letter menu to correct a binary selection would reintroduce the
            # A-lock on the very re-ask that exists to escape a bad pick.
            _re = _pick_file(_alt, [_disp_by[p] for p in _alt], "file_overflow")
            _file_selector_stage("file_overflow")
            cert["stages"].append({"insufficient_menu": {
                "reason": f"{fpick} contains no {cls} construct to fix",
                "offered": list(_alt), "picked": _re}})
            if _re is not None:
                fpick = _re
                cert["stages"].append({"file_by": "insufficient_menu_escape"})
    # ---- ANCHOR-SUPPORT REQUIREMENT FOR A FILE PICK (autopsy fix #3) ---------
    # THE CONFABULATION ATTRACTOR. On the real board a scrubbed symptom carries
    # zero content tokens, so BM25 ranks on the corpus's own term mass and the
    # file menu collapses onto the same few high-mass files regardless of the
    # task: `utils/grading_service.py` line 68 was proposed as the fix for THREE
    # unrelated tasks, `utils/prompt_composition.py` line 203 for two — votes
    # unanimous, margins up to 2.82. Five tasks, two distinct edits. A high
    # margin is not evidence; it is the model being confidently certain about a
    # file it has nothing to be certain from.
    #
    # So a MENU pick must be able to name its evidence. Three admissible kinds:
    #   * ANCHOR      the ticket named a function/class this file defines;
    #   * CONTENT     the ticket's own code-like tokens appear in this file;
    #   * STRUCTURAL  the file hosts the symptom class AND that is DISCRIMINATING
    #                 (a construct half the menu also has separates nothing — the
    #                 same "distinctive first" rule the evidence cards already
    #                 apply, and the same reason an inert witness may not
    #                 withdraw `none`).
    # With none of the three the menu had no evidentiary basis, and abstaining is
    # the honest answer: this converts a fabricated edit into a recorded no-pick.
    # The PICK IS STILL RECORDED so recall-vs-support stays separable in the
    # trace. Never applied to a pinned file (proof, not a choice) — same scoping
    # as the insufficient-menu escape.
    if (fpick is not None and not _pinned
            and getattr(cfg.locate, "require_file_support", True)):
        _anchor_paths = {p for p, _ in anchors}
        _support = []
        if fpick in _anchor_paths:
            _support.append("anchor")
        _fsrc = files.get(fpick, "")
        if any(re.search(rf"\b{re.escape(t)}\b", _fsrc) for t in code_like):
            _support.append("content_token")
        # GROUNDED-SYMPTOM CHANNEL (measured 2026-08-29).
        # The `code_like` channel above only sees snake_case/ClassName tokens, so a
        # ticket written in PROSE ("…around gate, action, approve and pair…") can
        # never produce content support even when its grounded nouns are all over
        # the file. Measured on the fresh-21 real board: the gate fired on all 21
        # tasks, found NO support on all 21, and abstained — while the model's file
        # pick was already the GOLD FILE on 16 of them. One gate, whole board.
        #
        # This channel asks the question the harvest guarantees an answer to: the
        # symptom's own content words (boilerplate stripped, the same stoplist the
        # ranker and the evidence cards use) matched against the file's identifier
        # vocabulary — and DISCRIMINATING, exactly as the structural channel is: a
        # word half the menu also carries separates nothing.
        # Calibration (measured, per-task, on the same 21): with the threshold at
        # 2 distinctive words all 16 correct picks are supported and 4 of the 5
        # CONFABULATED picks are still refused (utils/attempt_context.py scores 1
        # on every one of them — it is the real-19 attractor and it stays out).
        _cmin = int(getattr(cfg.locate, "file_support_content_min", 0) or 0)
        if _cmin:
            _cw = file_rank_content_words(task_text)
            _menu_paths = list(cand) or [fpick]
            _cw_hits = sorted(_cw & set(idx.get(fpick) or {}))
            _distinct = [w for w in _cw_hits
                         if sum(1 for q in _menu_paths
                                if w in (idx.get(q) or {})) * 2 <= len(_menu_paths)]
            cert["file_support_content"] = {"hits": _cw_hits[:12],
                                            "distinctive": _distinct[:12]}
            if len(_distinct) >= _cmin:
                _support.append("symptom_content")
        _structural_ok = True
        if cls and fxi is not None:
            try:
                _menu = list(cand) or [fpick]
                _hosts = [p for p in _menu
                          if (fxi._count_constructs(files.get(p, "")) or {}
                              ).get(cls, 0) > 0]
                if fpick in _hosts and len(_hosts) * 2 <= len(_menu):
                    _support.append("structural")
            except Exception as _e:
                _structural_ok = False
                cert.setdefault("warnings", []).append(
                    f"file-support structural check unavailable ({type(_e).__name__})")
        elif cls:
            _structural_ok = False         # the function index never ran
        cert["stages"].append({"file_support": {
            "path": fpick, "kinds": _support, "supported": bool(_support),
            "structural_measurable": _structural_ok}})
        # FAIL OPEN ON AN UNMEASURED CHANNEL. "No support" is only a finding when
        # every channel was actually evaluated; a structural check that could not
        # run makes the absence unverified, and an unverified absence must never
        # become an abstain (the same rule that forbids an absence claim made
        # through a tool that could not see).
        if not _support and not _structural_ok:
            cert["stages"].append({"file_support_unenforced":
                                   "structural channel unavailable"})
        elif not _support:
            cert["reason"] = "locate: file pick has no evidentiary support"
            cert["unsupported_file_pick"] = fpick
            cert["candidates"] = list(cand) + list(over_cand)
            return cert
    if fpick is None:
        cert["reason"] = "locate: file menu abstained"
        cert["candidates"] = list(cand) + list(over_cand)
        return cert
    cert["stages"].append({"file": fpick})
    src = files[fpick]
    # TOCTOU pre-image: the hash of exactly the bytes every downstream decision
    # (and the diff) was computed against. Re-verified immediately before write.
    src_sha = _sha256_text(src)
    tree = ast.parse(src)
    fns = [x.name for x in ast.walk(tree)
           if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # function-granularity dividend: the retrieval already ranked the repair
    # locus — its function leads the function menu.
    ev = func_evidence.get(fpick)
    best_fn = (ev.get("best_func") if isinstance(ev, dict) else None)
    if cfg.locate.function_rank_by_evidence and fidx is not None:
        # FULL evidence order, not just a single promotion: the live 1B answers
        # the function menu with 'A' nearly every time, so option order IS the
        # decision. Measured: gold is option A in 19/21 under this order vs 4/21
        # under AST-walk order.
        try:
            fns = fxi.rank_function_names(rank_task, cls, fidx, fpick, fns)
        except Exception as _e:
            cert.setdefault("warnings", []).append(
                f"function ranking unavailable ({type(_e).__name__})")
    elif best_fn:
        short = best_fn.split(".")[-1]
        if short in fns:
            fns = [short] + [f for f in fns if f != short]
    if not fns:
        cert["reason"] = f"{fpick} has no functions"
        return cert
    # C0: a task-named function defined in the picked file wins outright
    named_fns = [tok for p, tok in anchors if p == fpick and tok in fns]
    # DOCSTRING-VERBATIM PIN (rung-4000 autopsy, ranked fix #7). These tasks are
    # GENERATED from the target function's docstring, so a ticket that contains a
    # function's docstring verbatim has already named the function — it is the
    # same class of evidence as `anchor_forces_file` at the file stage, and it is
    # strictly harder evidence than the symptom-WORD COUNT the ranker uses.
    # MEASURED: `api/teacher_chat.py::create_session` was option B and the ticket
    # quotes its docstring; the menu voted unanimously for `_legacy_chat_stream`
    # on "6 symptom words vs 5" and the run spent its whole budget in the wrong
    # function. ONLY A UNIQUE MATCH PINS — uniqueness is what makes a pin safe
    # (the same law the symptom anchor obeys); two functions sharing a quoted
    # docstring is an ambiguity the menu should still resolve.
    doc_pin = None
    if getattr(cfg.locate, "docstring_function_pin", True):
        _quoted = [f for f in fns
                   if _task_quotes_docstring(
                       task_text,
                       next((x for x in ast.walk(tree)
                       if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and x.name == f),
                            None))]
        cert["docstring_quoted_functions"] = _quoted[:8]
        if len(_quoted) == 1:
            doc_pin = _quoted[0]
    if (sym_anchor is not None and sym_anchor[0] == fpick
            and sym_anchor[1] in fns and cfg.locate.symptom_anchors):
        # the same unique phrase->name match that pinned the file names the
        # function outright; the menu adds nothing a code match already proved.
        fnpick = sym_anchor[1]
        cert["stages"].append({"function_by": "symptom_anchor"})
    elif len(set(named_fns)) == 1:
        fnpick = named_fns[0]
        cert["stages"].append({"function_by": "anchor"})
    elif doc_pin is not None:
        # ORDERED BELOW BOTH ANCHORS ON PURPOSE: an exact symbol the ticket
        # SPELLED is harder evidence than a docstring it quoted, and the pin's
        # job is to beat the symptom-WORD-COUNT ranker (which is what lost
        # `create_session`), not to outrank a code match.
        fnpick = doc_pin
        cert["stages"].append({"function_by": "docstring_verbatim"})
    else:
        opts = fns[:26]
        disp = opts
        if cfg.locate.function_evidence and fidx is not None:
            # EVIDENCE, not prose: the menu used to show bare names, and the gym's
            # function.pick failures were files whose gold function already ranked
            # first by score. Options are unchanged (the pick maps back by name);
            # only the DISPLAY gains a computed evidence line.
            try:
                lines = fxi.function_evidence_lines(base_task, cls, fidx, fpick,
                                                    opts)
                # menu-wide, so a word every option shares sorts last
                fn_df = menu_doc_frequency([symptom_words_of(lines.get(o))
                                            for o in opts])
                fn_nodes = {x.name: x for x in ast.walk(tree)
                            if isinstance(x, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))}
                disp = [function_card(o, lines[o], fn_nodes.get(o), fn_df)
                        if o in lines else str(o) for o in opts]
                disp = distinguish_cards(opts, disp)
            except Exception as _e:      # evidence is additive, never fatal
                cert.setdefault("warnings", []).append(
                    f"function evidence unavailable ({type(_e).__name__})")
                disp = opts
        # ---- PRIMARY/OVERFLOW SPLIT (E100 task 2) ---------------------------
        # MEASURED BLOWER, 252c12de: all 26 candidates in ONE menu produced a
        # 7596-token prompt and 3/3 finish_reason='length'. The harness refused
        # the truncated completions — #11d working — and the runs could not
        # produce an answer at all. The file menu already solved this exact
        # shape: bound the FIRST menu, make the rest a SECOND one asked only
        # when the first produced no pick. Never a wider first chance.
        # Under the binary selector the split also bounds the BILL, since each
        # primary option is another probe call.
        _fn_primary_k = max(0, int(getattr(cfg.locate, "function_menu_primary_k",
                                           0) or 0))
        if _fn_primary_k and len(opts) > _fn_primary_k:
            # THE OVERFLOW MENU IS A MENU TOO: it obeys the same eight-choice
            # width contract as the primary, exactly as the file overflow does
            # (`room` in the locate stage). Handing the leftovers over in one
            # 18-wide menu would simply move the blower one call to the right.
            _fn_over_hi = _fn_primary_k + MAX_MODEL_FILE_MENU_OPTIONS
            fn_over = opts[_fn_primary_k:_fn_over_hi]
            fn_over_disp = disp[_fn_primary_k:_fn_over_hi]
            opts, disp = opts[:_fn_primary_k], disp[:_fn_primary_k]
        else:
            fn_over, fn_over_disp = [], []
        cert["function_overflow_candidates"] = list(fn_over)
        cert.setdefault("retrieval", {})["function_menu"] = (
            list(opts) + list(fn_over)
        )
        _fn_selector = str(getattr(cfg.fix, "function_selector", "letter")
                           or "letter")
        _fn_nodes_all = {x.name: x for x in ast.walk(tree)
                         if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def _pick_function(_opts, _disps, _label):
            """One function menu, by whichever selector is armed.

            Binary path: one independent yes/no probe per candidate function over
            its OWN body (`function_probe_body`, bounded and truncation-marked),
            code ranks, one inverted confirm may veto. Position is not an input,
            which is the whole point — the letter menu here answered 'A' nearly
            every time (see `function_rank_by_evidence`, which is a mitigation of
            the A-lock rather than a cure for it).

            The info-request wrapper stays letter-only: it offers
            `REQUEST(SHOW_SOURCE, B)`, and a binary probe has no lettered list to
            point at. Nothing is lost by it here — the probe is ALREADY shown the
            function's source, which is what a request would have fetched.
            """
            if _fn_selector == "binary" and _opts:
                _bodies = {o: function_probe_body(src, _fn_nodes_all.get(o))
                           for o in _opts}
                # same knob, same measurement, same reason as the file stage
                _cq = None
                if getattr(cfg.fix, "location_probe_confirm", False):
                    def _cq(o):
                        return FUNC_CONFIRM_Q.format(task=base_task, path=fpick,
                                                     fn=o, body=_bodies[o])
                return fx.menu_pick_binary(
                    list(_opts), list(_disps), _label,
                    lambda o: FUNC_PROBE_Q.format(task=base_task, path=fpick,
                                                  fn=o, body=_bodies[o]),
                    _cq,
                    probe_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"in_function": "<yes|no>"}.'),
                    confirm_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"unrelated": "<yes|no>"}.'))
            return fx.menu_pick(
                lambda menu: FUNC_MENU_Q.format(task=base_task, path=fpick,
                                                menu=menu),
                _opts, _disps, _label,
                rctx=RequestContext("function", files,
                                    lambda o, _p=fpick: {"path": _p,
                                                         "func": str(o)},
                                    cfg, path=fpick))

        def _fn_selector_stage(_label):
            _r = fx.margins[-1] if fx.margins else {}
            if (_fn_selector == "binary" and isinstance(_r, dict)
                    and _r.get("menu") == _label and _r.get("binary_probe")):
                cert["stages"].append({"function_selector": _r["binary_probe"]})

        fnpick = _pick_function(opts, disp, "function")
        _fn_selector_stage("function")
        if fnpick is None and fn_over:
            fnpick = _pick_function(fn_over, fn_over_disp, "function_overflow")
            _fn_selector_stage("function_overflow")
            if fnpick is not None:
                cert["stages"].append({"function_by": "menu_overflow"})
    if fnpick is None:
        cert["reason"] = "function menu abstained"
        return cert
    cert["stages"].append({"function": fnpick})
    # ---- FUNCTION SUPPORT GATE (pre-authoring) -----------------------------
    # A rank-1 function is not automatically a supported function. Compare the
    # selected body with the other offered bodies using only code-owned facts:
    # explicit identifiers, distinctive symptom terms, and class feasibility.
    # Absence alone stays UNKNOWN; only a materially stronger alternative can
    # reject the pick, which protects cross-module and comment-heavy fixes.
    _function_support_result = None
    if getattr(cfg.locate, "function_support_gate", True):
        try:
            _fn_nodes_for_gate = {
                x.name: x for x in ast.walk(tree)
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            _gate_lines = src.splitlines()
            _gate_sources = {}
            _gate_counts = {}
            for _name in dict.fromkeys(fns):
                _gate_node = _fn_nodes_for_gate.get(_name)
                if _gate_node is None:
                    continue
                _gate_lo = max(0, _gate_node.lineno - 1)
                _gate_hi = min(
                    len(_gate_lines),
                    getattr(_gate_node, "end_lineno", _gate_node.lineno),
                )
                _gate_source = "\n".join(_gate_lines[_gate_lo:_gate_hi])
                _gate_sources[_name] = _gate_source
                _gate_counts[_name] = 0
                try:
                    _gate_fxi = fxi
                    if _gate_fxi is None:
                        import func_index as _gate_fxi
                    _gate_counts[_name] = int(
                        (_gate_fxi._count_constructs(_gate_source) or {}).get(cls, 0)
                    )
                except Exception:
                    # Structural evidence is optional; lexical evidence remains
                    # valid and the receipt records no fabricated count.
                    _gate_counts[_name] = 0
            _pinned_functions = {
                fn for stage in cert["stages"]
                if isinstance(stage, dict)
                and stage.get("function_by") in {
                    "symptom_anchor", "anchor", "docstring_verbatim"
                }
                for fn in (fnpick,)
            }
            _function_evidence = _function_relevance.analyze_functions(
                task_text,
                _gate_sources,
                structural_counts=_gate_counts,
                symptom_terms=set(symptom_vocabulary(task_text)),
                pinned=_pinned_functions,
            )
            _function_support_result = _function_relevance.selection_gate(
                fnpick, _function_evidence
            )
            cert["stages"].append({
                "function_support": _function_support_result
            })
            if _function_support_result.get("decision") == "reject":
                cert["reason"] = (
                    "locate: selected function contradicted by stronger "
                    "deterministic function evidence"
                )
                cert["wrong_function"] = fnpick
                cert["candidates"] = list(fns)
                return cert
        except Exception as _e:
            # A safety sensor that cannot inspect the snapshot has no opinion;
            # it must not turn an analysis failure into a false rejection.
            cert.setdefault("warnings", []).append(
                f"function support unavailable ({type(_e).__name__})"
            )
            cert["stages"].append({
                "function_support": {
                    "decision": "unknown",
                    "reason": "function_support_sensor_unavailable",
                }
            })
    node = next(x for x in ast.walk(tree)
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                and x.name == fnpick)
    lines = src.splitlines()
    lo, hi = node.lineno - 1, getattr(node, "end_lineno", node.lineno)
    span = lines[lo:hi]
    # comment facts for the PICKED function, prepended to the fix-stage prompts
    cfacts = _comment_facts(cfg, fpick, src, fnpick, stats=io_stats)
    fix_task = ("COMMENT FACTS:\n" + "\n".join(f"  - {c}" for c in cfacts) +
                "\n\n" + base_task) if cfacts else base_task
    if cfacts:
        cert["comment_facts"] = cfacts
    new_src = None
    # --- FIX LADDER rung 1: mutation menu (closed candidates) ---
    fam_enabled = [f for f in cfg.fix.families if f in FAMILY_MAP]
    enabled_cls = {FAMILY_MAP[f] for f in fam_enabled}
    muts = []
    if cfg.fix.mutation_menu:
        # FILE CONTEXT for the enumerator (`wrong_attribute` only, and optional
        # by construction — every other family's output is byte-identical with or
        # without it). The measured need: the real-history gold `.message` ->
        # `.content` is only reachable when the attribute pool is drawn from the
        # WHOLE FILE rather than the one function, and with the pool routed the
        # menu is 8 options instead of 60.
        #
        # Routing — single-family / shape-preferred union / flat union — lives
        # in `build_mutation_menu`, so it can be measured offline without a
        # model call. Behaviour here is unchanged for every class that was
        # already routed.
        # CORPUS PROVIDER INDEX — the writer for the census tranche's reader
        # (#223). `c1_import_change` proposes `from M import N` for a name
        # NOTHING in the parent file binds; the parent file therefore cannot be
        # its evidence, and without this the family is structurally empty rather
        # than merely unlucky. `files` is the corpus `load_repo` already
        # collected for retrieval, so this costs one AST pass per file and no
        # new I/O. Best-effort: a failure here degrades the census tranche to a
        # file-local index, never the pipeline.
        providers = None
        try:
            providers = _edit_sets.build_provider_index(files, exclude=fpick)
        except Exception as _e:
            cert.setdefault("warnings", []).append(
                f"provider index unavailable ({type(_e).__name__})")
        muts = build_mutation_menu(span, lo, cls, lines, fam_enabled,
                                   providers=providers)
        if providers is not None:
            cert["provider_index"] = {
                "files_scanned": providers.files_scanned,
                "bindings": len(providers.imports),
                "truncated": providers.truncated,
            }
        # ORDERING DEFECT, FIXED 2026-08-29 (mech-lane autopsy). This used to be
        # `muts = muts[:cfg.fix.menu_cap]` — the menu was TRUNCATED here and
        # only SORTED 100 lines below, at the `evidence_ranked` block, so a
        # candidate the ranking would have promoted to position 1 was evicted
        # before the ranking ever saw it. Measured shape of the loss: the
        # ea639fc1 `.message` -> `.content` gold sits at index 42 of 61 in the
        # enumeration and is unreachable behind an 8-wide cap taken here, no
        # matter how strong its evidence is.
        #
        # RANK FIRST, THEN CAP. What remains here is an EXPLORATION ceiling, not
        # the menu: it exists so a pathological union enumeration cannot make
        # the ranking O(huge). With `evidence_ranked` off the final
        # `muts[:menu_cap]` below yields the identical first-N, so this is a
        # no-op for every unranked path.
        muts = muts[:max(cfg.fix.menu_cap, MENU_EXPLORE_CAP)]
        # --- TYPED-HOLE TRANCHE (holes.py) --------------------------------
        # THE NOVEL-TOKEN BOUNDARY. Measured on the v3 real board: 16 of 21
        # golds contain at least one token that occurs NOWHERE in the parent
        # file — invented locals, authored prose, corpus-unique names. Every
        # enumerator here can only re-propose tokens the parent already has, so
        # `PUNTED_PATTERNS` and `CENSUS_PUNTS` classify those misses NOVEL-TOKEN:
        # a hard expressiveness boundary, not a pruning failure.
        #
        # The tranche appends candidates whose STRUCTURE was enumerated
        # deterministically and whose one typed slot the model filled. It is
        # APPENDED, out of its own budget, after `muts` is already final — the
        # same discipline the census tranche landed under — so it can neither
        # reorder nor evict an existing candidate here. (It CAN change which
        # candidates survive `menu_cap` after `evidence_ranked` sorts, exactly
        # as any added family does; that is documented on `FixCfg.typed_holes`
        # and pinned by test_typed_holes_offline.py.)
        #
        # A filled candidate is an ordinary EditSet. It carries no priority and
        # no bypass: prose filter, inert filter, evidence rank, spec_exec,
        # witness and ballot all see it on exactly the terms they see an
        # enumerated one. That is the whole safety argument.
        if getattr(cfg.fix, "typed_holes", False):
            _h_tel = _typed_hole_tranche(span, lo, lines, muts, cfg, cert)
            if _h_tel.get("candidates"):
                muts = muts + _h_tel.pop("_candidates")
            else:
                _h_tel.pop("_candidates", None)
            cert["stages"].append({"typed_holes": _h_tel})
            cert["typed_holes"] = _h_tel
        # --- BLOCK-HOLE TRANCHE (block_holes.py) ---------------------------
        # THE MULTI-STATEMENT BOUNDARY. Measured on the v3 real board: 14 of the
        # 21 golds are not single-block ADDITIONS — they add statements AND
        # rewrite or delete neighbouring ones. No single-line family and no
        # block TEMPLATE can express that; a contiguous-run REPLACEMENT can, and
        # that is the shape this family enumerates alongside insertion.
        #
        # ON for the configured expressible shapes, under the fail-open doctrine.
        # The oracle ceiling is 10/21 (probes/block_hole_reachability.py) and
        # NOTHING here bypasses arbitration: a block candidate is an ordinary
        # `EditSet` that the prose filter, inert filter, evidence rank, spec_exec,
        # witness replay and every gate see on exactly the terms they see an
        # enumerated one. Shape-gating avoids spending fills on one-line routes
        # such as `attribute_swap` while leaving the measured block shapes live.
        #
        # APPENDED after `muts` is final, out of its OWN call budget, exactly as
        # the typed-hole tranche is — it can neither reorder nor evict an
        # existing candidate at this point. Menu pressure (screen 8 / expose 2)
        # is applied INSIDE the tranche; see `_block_hole_tranche`.
        if getattr(cfg.fix, "block_hole", True):
            if _block_hole_shape_allowed(cls, cfg):
                _b_tel = _block_hole_tranche(
                    span, lo, lines, cfg, cert,
                    symptom=task_text, fn_name=fnpick)
            else:
                _b_tel = {
                    "enabled": True,
                    "skipped": True,
                    "reason": "shape_not_allowlisted",
                    "shape": cls,
                }
            if _b_tel.get("candidates"):
                muts = muts + _b_tel.pop("_candidates")
            else:
                _b_tel.pop("_candidates", None)
            cert["stages"].append({"block_holes": _b_tel})
            cert["block_holes"] = _b_tel
    # WRONG-ATTRIBUTE family (smoke-1: the real fix was .message -> .content and
    # no family could express it): when the task mentions .attr, and a line in the
    # function reads that attr, candidates = swap to the REAL attributes of any
    # task-named class (grounded in the class's own source — name-grounding as a
    # fix generator).
    if cfg.fix.mutation_menu and "attr" in cfg.fix.families:
        for cls_name in {tok for _, tok in anchors if tok[:1].isupper()} | \
                        {t for t in code_like if t[:1].isupper()}:
            cls_src = None
            for p2, s2 in files.items():
                m2 = re.search(rf"\bclass {re.escape(cls_name)}\b", s2)
                if m2:
                    cls_src = s2
                    break
            if cls_src is None:
                continue
            real_attrs = set(re.findall(rf"^\s+(\w+)\s*[:=]", cls_src, flags=re.M))
            # instance attributes only — ALL_CAPS are class constants (smoke-5
            # swapped to .ASSISTANT, a role constant)
            if cfg.fix.exclude_const_attrs:
                real_attrs = {a for a in real_attrs if not a.isupper()}
            for bad in attr_mentions:
                if bad in real_attrs:
                    continue
                for off, line in enumerate(span):
                    if f".{bad}" in line:
                        # a task-MENTIONED attribute is the answer being spelled out
                        # for us — those candidates go first, alone if possible
                        mentioned = [g for g in sorted(real_attrs)
                                     if g in attr_mentions and g != bad]
                        rest = [g for g in sorted(real_attrs)
                                if g not in attr_mentions and not g.startswith("_")
                                and g != bad]
                        order = (mentioned + rest) if cfg.fix.mentioned_attr_first \
                            else sorted(set(mentioned + rest))
                        for good in order:
                            muts.append((lo + off, line,
                                         line.replace(f".{bad}", f".{good}")))
    # INERT-CANDIDATE FILTER (the
    # model picked a comment-prose edit — "struggled (low score or >1 attempt)"
    # -> "... and >1 attempt" — which compiles, reads plausibly, and changes NO
    # behavior at all, burning the whole task). Drop candidates whose textual
    # delta touches no executable token. Fail OPEN: a candidate we cannot
    # tokenize survives. mutations.py's enumerator is untouched on purpose
    # (harvest tests need prose mutations discoverable).
    #
    # PROSE EXCLUSION (real-19 autopsy fix #4) is the SECOND half of the same
    # filter and it is the half the delta test cannot do. `_is_inert_change`
    # tokenizes one line on its own, so a line lifted out of a triple-quoted
    # docstring reads as ordinary NAMEs and its `and` -> `or` looks executable.
    # `prose_regions` tokenizes the WHOLE FILE, so the interior of a multi-line
    # string is known for what it is. Measured cost of not having this: 2 of the
    # 3 FIX_PROPOSED-wrong rows on the real board were prose edits (a docstring
    # and an LLM prompt string), each of which passed every existing gate.
    # Multi-line candidates are exempt by construction — they insert statements.
    if muts and getattr(cfg.fix, "exclude_prose_candidates", True):
        _regions = prose_regions(src)
        _kept = [m for m in muts if not _is_prose_edit(_regions, m[0], m[1], m[2])]
        _pdropped = len(muts) - len(_kept)
        if _pdropped:
            # FAIL SAFE, NOT OPEN: unlike the inert filter, an all-prose menu is
            # EMPTIED. A menu whose every option is a comment edit has no repair
            # on it, and an honest "no admissible candidate" beats shipping the
            # one that reads best. Never charge a task for a prose fix.
            # HONEST INTERACTION, NOT A CARVE-OUT. A `hole_literal_value`
            # candidate changes text INSIDE a string literal, so this filter
            # drops it — and it SHOULD, because the filter's measurement (2 of
            # 3 wrong FIX_PROPOSED rows on the real board were prose edits) is
            # older and stronger than the typed-hole family. The tension is
            # real: v3 rows 8 and 9 are `literal_change` golds that edit an LLM
            # prompt string, so this gate is also what makes them unreachable.
            # That is reported here rather than resolved by exempting the new
            # family, which would be weakening a gate to make a new feature
            # look good. `probes/hole_reachability.py` prints the same number.
            _hole_dropped = sum(1 for m in muts
                                if getattr(m, "from_hole", False)
                                and m not in _kept)
            muts = _kept
            cert["stages"].append({"prose_candidates_dropped": _pdropped,
                                   "remaining": len(_kept),
                                   "typed_hole_candidates_dropped":
                                       _hole_dropped})
            cert["prose_candidates_dropped"] = _pdropped
            if _hole_dropped:
                cert.setdefault("typed_holes", {})[
                    "dropped_by_prose_filter"] = _hole_dropped
    if muts:
        # UNCALLABLE-CALLEE SCREEN (task 2b5d3f3a, 2026-08-29). A candidate that
        # turns a builtin callee into a local that is never called anywhere in
        # the function raises TypeError the first time its branch runs. It
        # compiles, so it passed every existing gate and shipped as a confident
        # FIX_PROPOSED scoring fixed=0 -- a false green. Screened here, with the
        # count reported separately from the prose screen so the two mechanisms
        # stay attributable.
        _uncallable = [m for m in muts
                       if uncallable_call_target(span, m[1], m[2])]
        if _uncallable and len(_uncallable) < len(muts):
            _reasons = sorted({uncallable_call_target(span, m[1], m[2])
                               for m in _uncallable})
            muts = [m for m in muts if m not in _uncallable]
            cert["stages"].append({"uncallable_candidates_dropped":
                                   len(_uncallable), "reasons": _reasons[:4]})
        kept = [m for m in muts if not _is_inert_change(m[1], m[2])]
        dropped = len(muts) - len(kept)
        if dropped and kept:
            muts = kept
            cert["stages"].append({"inert_candidates_dropped": dropped})
            cert["inert_candidates_dropped"] = dropped
        elif dropped:
            # every candidate was prose — keep the list rather than empty the
            # ladder rung; the count is still reported honestly.
            cert["inert_candidates_dropped"] = 0
            cert.setdefault("warnings", []).append(
                f"all {dropped} mutation candidates were inert (kept, not dropped)")
    # Evidence-ranked candidate ordering (smoke-3: the .message->.content candidate
    # sat below the cap while a nearby cmp-flip got picked). A candidate whose
    # CHANGED text overlaps the task's own symbols outranks everything.
    task_syms = attr_mentions | {t.lower() for t in code_like} | words(task_text)

    def evidence(mut):
        li, old, new = mut
        # strongest: the INTRODUCED attribute is one the task itself mentions
        new_attrs = set(re.findall(r"\.([a-z_][a-z0-9_]{2,})", new)) - \
                    set(re.findall(r"\.([a-z_][a-z0-9_]{2,})", old))
        delta = set(re.findall(r"[a-z_]{3,}", (old + " " + new).lower()))
        changed_toks = set(re.findall(r"[a-z_]{3,}", new.lower())) - \
                       set(re.findall(r"[a-z_]{3,}", old.lower()))
        return (len(new_attrs & attr_mentions),
                len(changed_toks & task_syms), len(delta & task_syms))

    if cfg.fix.evidence_ranked:
        muts.sort(key=evidence, reverse=True)
    muts = muts[:cfg.fix.menu_cap]
    cert["stages"].append({"mutations": len(muts)})
    # --- INTENT ORACLE (intent_oracle.py) — the pamphlet on the door ----------
    # ARMED 2026-08-29. Two halves, both behind facts.intent_oracle:
    #
    #   FILTER   `discriminate` mines EXECUTABLE expectations from the function's
    #            own docstring + signature (zero model calls) and kills the
    #            candidates that provably contradict the stated contract.
    #   VERDICT  `render_evidence` renders the SAME judgement as one short line
    #            per surviving candidate, which is APPENDED to that candidate's
    #            mutation-menu option text.
    #
    # The confirmation stage is essential: a
    # filter that silently drops options teaches the model nothing, while the
    # same verdict written ON the option turns a positional guess into a reading
    # task. The model sees "passes docstring check: ..." / "FAILS docstring
    # check: ..." beside each correction.
    #
    # APPEND-ONLY, NON-NEGOTIABLE: option ORDER is untouched and option TEXT is
    # only suffixed with " | <verdict>". Verdicts are keyed by candidate IDENTITY
    # (id()), so they survive every later filter and reorder — witness replay
    # narrows `muts` to survivors and the line ballot re-partitions it, and both
    # preserve the tuple objects, so each option keeps its own verdict.
    # "no signal" verdicts are not rendered: a suffix that says nothing is noise
    # on every option and would dilute the two that carry a real judgement.
    #
    # P7 — NEVER SILENT. The armed cycle recorded `facts.intent_oracle: true` in
    # all 21 config blobs and emitted ZERO intent_oracle stages, because the
    # stage was only appended on the path where the block executed. A knob ON in
    # the ledger with no stage in the ledger is indistinguishable from a knob ON
    # with a dead consumer — the #181c silent-drop family, one layer up. So the
    # stage is emitted UNCONDITIONALLY whenever the knob is on, carrying a
    # `reason` when it did nothing.
    oracle_verdicts = {}                 # id(mut) -> verdict line
    if getattr(cfg.facts, "intent_oracle", False) and not muts:
        cert["stages"].append({"intent_oracle": {
            "applied": False, "reason": "no mutation candidates to discriminate",
            "candidates": 0, "survivors": 0, "expectations": 0,
            "killed": 0, "verdict_lines": 0}})
    if muts and getattr(cfg.facts, "intent_oracle", False):
        try:
            import textwrap as _tw

            import intent_oracle as _io
            _fn_src = _tw.dedent("\n".join(span))
            _res = _io.discriminate(muts, _fn_src, span)
            _lines = _io.render_evidence(_res)
            _surv = {id(c) for c in (_res.get("survivors") or [])}
            _pairs = list(zip(_res.get("candidates") or [], _lines))
            for _cand, _line in _pairs:
                if _line and _line != "no signal":
                    oracle_verdicts[id(_cand)] = _line
            _kept = [m for m in muts if id(m) in _surv]
            _killed = len(muts) - len(_kept)
            if _kept and _killed:
                muts = _kept
            # Prune verdicts for candidates that are gone. `id()` is only unique
            # among LIVE objects, and a killed tuple's id can be recycled — an
            # entry nothing will look up is still a stale key waiting to alias.
            _live = {id(m) for m in muts}
            oracle_verdicts = {k: v for k, v in oracle_verdicts.items()
                               if k in _live}
            cert["stages"].append({"intent_oracle": {
                "expectations": len(_res.get("expectations") or []),
                "candidates": len(_pairs),
                "killed": _killed if _kept else 0,
                "survivors": len(muts),
                "verdict_lines": sum(1 for v in oracle_verdicts.values() if v),
                "applied": bool(_res.get("applied")),
                "reason": ("ok" if _res.get("applied") else
                           "no expectations mined from the docstring/signature"
                           if not (_res.get("expectations") or [])
                           else "expectations decided nothing"),
            }})
        except Exception as _e:          # oracle is additive, never fatal
            oracle_verdicts = {}
            cert["stages"].append({"intent_oracle": {
                "applied": False,
                "reason": f"raised {type(_e).__name__}",
                "candidates": len(muts), "survivors": len(muts),
                "expectations": 0, "killed": 0, "verdict_lines": 0}})
            cert.setdefault("warnings", []).append(
                f"intent oracle unavailable ({type(_e).__name__})")
    # --- WITNESS REPLAY -------------------------
    # The measured wobble lives in vague-symptom BOOLEAN menus, where the 1B's
    # top-2 margin collapses and the pick flips across backends. Most of that
    # menu is answerable by CODE: evaluate every candidate predicate against
    # derived boundary witnesses, eliminate the ones provably wrong (no behavior
    # change at all, or an edit shape that contradicts the symptom class), and if
    # exactly ONE evaluable candidate survives, select it outright. The model is
    # only ever asked to break a genuine tie. Fail-open at every step: anything
    # unevaluable survives, and zero survivors leaves the original list standing.
    wr_pick, wr_ev = None, None
    wr_supported_survivors = 0     # gates the free 'none of these' escape below
    if muts and getattr(cfg.fix, "witness_replay", False):
        from witness_replay import filter_candidates
        survivors, wr_evidence = filter_candidates(task_text, span, muts)
        cert["stages"].append({"witness_replay": {
            "candidates": len(muts),
            "survivors": len(survivors),
            "supported": sum(1 for e in wr_evidence if e["supported"]),
            "symptom_class": wr_evidence[0]["symptom_class"] if wr_evidence else None,
            "eliminated": [e["reason"] for e in wr_evidence if not e["survived"]][:8],
        }})
        if survivors:
            surv_ev = [e for e in wr_evidence if e["survived"]]
            # A STAGE THAT DISCRIMINATED NOTHING MAY NOT REMOVE THE ESCAPE HATCH
            # (real-19 autopsy fix #5). Measured: witness replay eliminated zero
            # candidates on 19 of 19 real runs, and on 6 of them its vacuous
            # "support" still withdrew the `none of these` option — the kernel
            # was forced to pick from a menu code had said nothing about. Support
            # is only evidence that a real fix is on this menu if the replay
            # actually separated candidates, so it is honoured ONLY when the
            # stage eliminated at least one.
            _wr_eliminated = sum(1 for e in wr_evidence if not e["survived"])
            _supported = sum(1 for e in surv_ev if e.get("supported"))
            wr_supported_survivors = _supported if _wr_eliminated > 0 else 0
            if _supported and not _wr_eliminated:
                cert["stages"].append({"witness_support_ignored": {
                    "supported": _supported, "eliminated": 0,
                    "reason": "witness replay discriminated nothing; its support "
                              "may not withdraw the 'none of these' escape"}})
            if len(survivors) == 1 and surv_ev[0]["supported"] \
                    and surv_ev[0]["rows_flipped"] > 0:
                wr_pick, wr_ev = survivors[0], surv_ev[0]
            muts = survivors
    if wr_pick is not None:
        li, _, newline = wr_pick
        # APPLY SEAM (edit_sets.FIXER_SEAM step 3). Byte-identical to the former
        # `c2 = list(lines); c2[li] = newline` for a plain 3-tuple; a coordinated
        # `EditSet` is spliced as ONE atomic block with its exact-match anchor
        # re-checked. None = the anchor drifted -> fail closed, never partial.
        c2 = _edit_sets.apply_candidate(lines, wr_pick)
        if c2 is None:                # never silent: an unanchorable pick is a
            cert.setdefault("warnings", []).append(   # fact, not a non-event
                "witness_replay pick could not be anchored (line %d)" % (li + 1))
    if wr_pick is not None and c2 is not None:
        try:
            compile("\n".join(c2), fpick, "exec")
            new_src = "\n".join(c2) + "\n"
            cert["stages"].append({
                "fix": "witness_replay", "function": fnpick,
                "line": li + 1,
                "line_span": _cert_line_span(wr_pick, li + 1),
                "change": _cert_change(wr_pick[1], newline),
                # RAW pair, for consumers that must TOKENIZE rather than read.
                # `change` is flattened for humans and joins old to new with
                # " -> ", which is not a parseable separator (a line may contain
                # one). A channel that split the display string would be reading
                # a presentation format as a contract.
                "change_old": wr_pick[1], "change_new": newline,
                "evidence": wr_ev})
        except SyntaxError:
            pass                      # fall through to the model menu
    # --- SPEC-EXEC ARBITER (spec_exec.py, wiring contract in its docstring) ---
    # The mutation menu's model call is a POSITION READOUT (E49). When the
    # candidate set is CLOSED — and by this point it is: the enumerators are
    # exhaustive, the inert filter and witness replay only REMOVE — we do not
    # have to ask. spec_exec applies every candidate in isolation and runs the
    # deterministic battery (compile / witness / discriminating test / symptom
    # direction). Survival picks.
    #
    #   DECIDED   -> take the candidate. ZERO model calls: this sets `new_src`,
    #                and the line ballot, the staged-reasoning call and the menu
    #                itself are all gated on `new_src is None`, so none of them
    #                fire. The apply-safety gates below still run in full — the
    #                arbiter picks a candidate, it does not authorise a write.
    #   SHORTLIST -> `muts` narrows to the survivors and everything downstream
    #                runs unchanged on the shorter menu (relative order kept).
    #   NO_SIGNAL -> nothing moves; `muts` is byte-identical.
    #
    # Placed AFTER witness replay because a unique witness survivor has already
    # settled the repair without a call, and BEFORE the ballot because the point
    # is to spend no calls at all on a menu code can answer. Append-only: the
    # certificate stage records every candidate, every elimination WITH the check
    # that killed it, and the evidence behind the verdict.
    if (muts and new_src is None and len(muts) > 1
            and getattr(cfg.fix, "spec_exec", False)):
        try:
            import spec_exec as _sx_mod
            import witness_replay as _wr_mod
            _sx = _sx_mod.arbitrate("\n".join(span), muts,
                                    _wr_mod.infer_symptom_class(task_text), span)
            cert["stages"].append(_sx.as_cert_stage())
            if _sx.verdict == _sx_mod.DECIDED and _sx.selected is not None:
                _li, _, _newline = _sx.selected
                _c2 = _edit_sets.apply_candidate(lines, _sx.selected)  # SEAM
                try:
                    if _c2 is None:      # anchor drifted: fail closed
                        raise SyntaxError("candidate could not be anchored")
                    compile("\n".join(_c2), fpick, "exec")
                    new_src = "\n".join(_c2) + "\n"
                    cert["stages"].append({
                        "fix": "spec_exec", "function": fnpick,
                        "line": _li + 1,
                        "line_span": _cert_line_span(_sx.selected, _li + 1),
                        "change": _cert_change(_sx.selected[1], _newline),
                        "change_old": _sx.selected[1], "change_new": _newline,
                        "evidence": {"affirmative": list(_sx.affirmative),
                                     "cases_cited": list(_sx.cases_cited),
                                     "testgen_verdict": _sx.testgen_verdict,
                                     "eliminated": len(_sx.eliminated),
                                     "reason": _sx.reason}})
                except SyntaxError:
                    pass              # fall through to the model menu
            elif _sx.verdict == _sx_mod.SHORTLIST and _sx.menu_candidates:
                muts = list(_sx.menu_candidates)
        except Exception as _e:       # the arbiter is additive, never fatal
            cert.setdefault("warnings", []).append(
                f"spec exec unavailable ({type(_e).__name__})")
    # --- FIX LADDER rung 1b: POSITION-FREE LINE BALLOT ------------------------
    # Decomposes the positional "which correction?" menu into independent
    # single-line yes/no questions (see LINE_BALLOT_Q for the measurement that
    # motivates it). The ballot NEVER selects a repair and never bypasses a gate:
    # its only power is to REORDER the menu below so the winning line's
    # candidates come first, because option order is empirically what the 1B's
    # answer tracks. The mutation menu, its margin probe and reversed retry, and
    # the compile and evidence-linkage gates all run afterwards unchanged.
    #
    # Placed AFTER witness replay on purpose: a unique witness survivor already
    # settled the repair without a model call (`new_src` set), and witness
    # elimination can collapse the candidates onto one line — both cases make the
    # ballot moot, so running here spends strictly fewer calls than running it
    # before the replay, with identical information.
    #
    # FAIL-SAFE BY CONSTRUCTION: the menu is reordered only when the ballot is
    # DECISIVE, i.e. the winner beats the line the menu currently leads with by
    # at least `line_ballot_tau`. Anything else (one distinct line, unparseable
    # answers, a saturated ballot where several lines tie at p(yes)=1.0, or the
    # winner already leading) leaves `muts` byte-identical to today's order. On
    # the 18-task measured set that guard produced 3 conversions and 0
    # regressions at tau=0.3; the reversed-polarity and paired variants scored
    # strictly worse for twice the calls and were dropped on that evidence.
    if muts and new_src is None and getattr(cfg.fix, "line_ballot", False):
        ordered_lines = []
        for _li, _o, _n in muts:
            if _li not in ordered_lines:
                ordered_lines.append(_li)
        # RUN-LOCAL budget: cfg default unless the attention allocator set one.
        budget = max(0, int(ballot_max_calls))
        probe_lines = ordered_lines[:budget]
        # A ballot that cannot compare the incumbent against a rival decides
        # nothing, so do not spend calls on it.
        if len(probe_lines) >= 2:
            first_line = {}
            for _li, _old, _n in muts:
                first_line.setdefault(_li, _old)
            tau = float(getattr(cfg.fix, "line_ballot_tau", 0.30))
            scores, calls = {}, 0
            for _li in probe_lines:
                try:
                    ballot_prompt = LINE_BALLOT_Q.format(
                        task=fix_task, fn=fnpick,
                        context=_line_ballot_context(
                            lines, lo, hi, _li,
                            max_lines=getattr(
                                cfg.fix, "line_ballot_context_lines", 11),
                            max_chars=getattr(
                                cfg.fix, "line_ballot_context_chars", 900)),
                        candidate_id="line_%d" % (_li + 1),
                        line=_truncate_for_prompt(
                            str(first_line[_li]).strip(), 120))
                    _t, _r = _call(
                        ballot_prompt, seed=cfg.seed, url=cfg.url, n_probs=20,
                        max_tokens=budget_for_stage(
                            cfg, "line_ballot", ballot_prompt),
                        enable_thinking=stage_thinking(cfg, "line_ballot"),
                        label="line_ballot",
                        continuation=CONTINUATION_LINE_BALLOT_Q)
                    calls += 1
                    s = _yes_no_score(_r)
                except Exception:
                    s = None            # fail OPEN: a dead probe just abstains
                if s is not None:
                    scores[_li] = s
            rec = {"line_ballot": {
                "calls": calls,
                "probed_lines": len(probe_lines),
                "distinct_lines": len(ordered_lines),
                "scores": {str(k): round(v, 4) for k, v in scores.items()},
                "incumbent_line": ordered_lines[0] + 1}}
            verdict = ballot_verdict(scores, ordered_lines[0], cfg)
            rec["line_ballot"].update(verdict)
            act, win = verdict["decision"], verdict.get("winner")
            if act.startswith("selected") and win is not None:
                _sel = [m for m in muts if m[0] == win]
                if _sel:
                    rec["line_ballot"]["dropped_other_lines"] = len(muts) - len(_sel)
                    muts = _sel
                else:                     # unreachable: win came out of `muts`
                    rec["line_ballot"]["decision"] = "kept: winner had no variants"
            elif act == "reordered" and win is not None:
                # stable partition: nothing is dropped, only re-ranked
                muts = ([m for m in muts if m[0] == win] +
                        [m for m in muts if m[0] != win])
            rec["line_ballot"].pop("winner", None)
            cert["stages"].append(rec)
    # Staged reasoning
    # ONE reasoning-only call before the mutation menu. It selects nothing: its
    # sanitized output is pasted into the menu prompt as PRIOR ANALYSIS. The
    # ladder autopsy's C5 traces show this 1B reasoning its way to the correct
    # line and then failing to name it in a parseable shape (27/84 runs); this
    # stage gives that reasoning somewhere to LAND, where the next call reads it.
    #
    # SANITIZED before it re-enters a prompt: brace literals are stripped (the
    # standing rule — they hang decisions downstream), whitespace collapsed, and
    # the whole thing capped. Model output is untrusted input to the next prompt.
    # FAIL-OPEN: any error leaves `mut_task` byte-identical to `fix_task`.
    mut_task = fix_task
    if muts and new_src is None and getattr(cfg.fix, "staged_reasoning", False):
        try:
            _first = {}
            for _li, _old, _n in muts:
                _first.setdefault(_li, _old)
            _lines_block = "\n".join(
                f"  line {k + 1}: {_truncate_for_prompt(str(v).strip(), 100)}"
                                     for k, v in _first.items())
            staged_prompt = STAGED_Q.format(task=fix_task, fn=fnpick,
                                            lines=_lines_block)
            _txt, _ = _call(
                staged_prompt, seed=cfg.seed, url=cfg.url,
                max_tokens=budget_for_stage(cfg, "staged_reasoning",
                                            staged_prompt),
                enable_thinking=stage_thinking(cfg, "staged_reasoning"),
                label="staged_reasoning",
                continuation=CONTINUATION_STAGED_Q)
            _cap = int(getattr(cfg.fix, "staged_reasoning_max_chars", 300))
            # SANITIZE: brace literals out (standing rule), and any `"pick": "X"`
            # directive the reasoning spelled out is REMOVED — the staged stage
            # is allowed to argue, never to vote. Leaving one in lets the same
            # call both advise the menu and answer it.
            _clean = re.sub(r'"?\bpick"?\s*:\s*"?[A-Za-z]"?', "", _txt or "")
            _clean = re.sub(r"\s+", " ", re.sub(r"[{}]", "", _clean)).strip()
            _raw_len = len(_clean)
            if _raw_len > _cap:
                # The marker is part of the public bounded value, so reserve
                # space for it before sentence completion. Appending it after
                # a cap-sized prefix violated the configured upper bound.
                _marker_budget = max(0, _cap - len(_TRUNCATION_MARKER) - 1)
                _capped_clean = _clean[:_marker_budget]
                _clean = _complete_sentences(_capped_clean)
                if _clean:
                    _clean += " " + _TRUNCATION_MARKER
                else:
                    _clean = _TRUNCATION_MARKER[:max(0, _cap)]
            else:
                _clean = _complete_sentences(_clean[:max(0, _cap)])
            if _clean:
                mut_task = fix_task + "\n\n" + STAGED_BLOCK_HEADER + _clean
                cert["staged"] = True
                cert["staged_analysis"] = _clean
                cert["stages"].append({"staged_reasoning": {
                    "chars": len(_clean), "lines_shown": len(_first),
                    "raw_chars": _raw_len, "cap": _cap,
                    "sentence_trimmed": len(_clean) < min(_raw_len, _cap)}})
            else:
                # NOTHING is better than an amputated clause (fix #4). Recorded
                # as its own outcome so "the stage produced no usable sentence"
                # never reads as "the stage did not run".
                cert["stages"].append({"staged_reasoning": {
                    "forwarded": False, "raw_chars": _raw_len, "cap": _cap,
                    "reason": "no complete sentence within the cap"}
                    if _raw_len else "empty"})
        except Exception as _e:          # staged reasoning is additive
            cert.setdefault("warnings", []).append(
                f"staged reasoning unavailable ({type(_e).__name__})")
    if muts and new_src is None:
        opts = list(muts)
        # A guard-insertion candidate's new_line carries embedded newlines (the
        # multi-line encoding in mutations.py). FLATTEN for display only: a raw
        # newline inside an option would split the "A) ..." menu the letter parser
        # depends on. The candidate tuple itself is untouched.
        # BLOCK CANDIDATES GET A SEMANTIC CARD, NOT A FLATTENED DIFF.
        # A block's `new` is many statements; flattened with " / " it renders as
        # the longest option on the menu by a wide margin, and option length is a
        # measured bias channel on this 1B (the binary probe already records
        # `length_chars` per candidate for exactly that reason). What the model
        # is shown instead is ONE sentence generated from the canonical role +
        # site metadata by `block_probe_phrase` — never from the model, never
        # from a comment. Presentation only: the phrase cannot upgrade evidence
        # or ranking, and the full patch is in the certificate's diff either way.
        base_displays = [
            (f"line {li+1}: {getattr(m, 'probe_phrase')}"
             if getattr(m, "probe_phrase", None)
             else f"line {li+1}: {_truncate_for_prompt(old.strip(), 58)}  ->  "
                  f"{_truncate_for_prompt(' / '.join(l.strip() for l in new.strip().split(chr(10))), 58)}")
            for m in muts for li, old, new in (m,)]
        # INTENT-ORACLE VERDICT LINES — append-only. Option order is unchanged
        # and each option's text is only SUFFIXED, so this can never move, drop
        # or add a candidate. See the oracle block above for why the verdict is
        # shown rather than silently enforced.
        menu_displays = [d + (" | " + oracle_verdicts[id(m)]
                              if id(m) in oracle_verdicts else "")
                         for d, m in zip(base_displays, muts)]
        displays = list(menu_displays)
        if oracle_verdicts:
            cert["intent_oracle_verdicts"] = [
                oracle_verdicts.get(id(m), "") for m in muts]
        # 'NONE' MUST EARN ITS PLACE (armed-cycle correction). `('none',None,None)`
        # is a free, always-available escape that costs exactly what a real
        # answer costs — and math_answer_verifier::_build_equivalent_forms took
        # it over a menu CONTAINING the gold fix, justifying it with a contract
        # the code does not have ("comparing a float to an integer ... greater
        # than or equal"). When witness replay has positively SUPPORTED at least
        # one surviving candidate, code has already established that a real fix
        # is on this menu, so "none of these" is not an available truth and the
        # option is withdrawn. With no supported survivor the escape stands: an
        # honest ABSTAIN beats a forced pick.
        none_withdrawn = bool(cfg.fix.none_option and wr_supported_survivors)
        if cfg.fix.none_option and not none_withdrawn:
            opts = opts + [("none", None, None)]
            displays = displays + ["none of these fixes the symptom"]
        if none_withdrawn:
            cert["stages"].append({"none_option": {
                "withdrawn": True,
                "reason": "witness replay supports %d surviving candidate(s); "
                          "'none of these' is not an available truth"
                          % wr_supported_survivors}})
        _mut_prompt = lambda menu: MUT_MENU_Q.format(task=mut_task, fn=fnpick,
                                                     menu=menu)
        # INFO-REQUEST context for the mutation/line menus: an option is a
        # (line_index, old, new) candidate, so SHOW_MORE_CONTEXT lands on the
        # lines AROUND the proposed edit — the exact hunger the E60 traces show
        # on mutation menus. The 'none of these' escape locates to nothing and
        # is therefore unfulfillable-but-charged, which is correct: asking to
        # see the source of "none" is a wasted question.
        def _mut_locate(opt, _p=fpick, _f=fnpick):
            if not isinstance(opt, tuple) or opt[0] == "none":
                return None
            return {"path": _p, "func": _f, "line": int(opt[0])}

        _mut_rctx = RequestContext("mutation", files, _mut_locate, cfg, path=fpick)
        # RETRY-SKIP ADMISSION (autopsy fix #1a). `muts` at this point IS the
        # post-witness survivor list, so admission is membership in it — and
        # only when witness replay actually RAN, because an ungated menu must
        # not be able to claim a gate it never faced. Identity, not equality:
        # the same tuple objects travel through every filter and reorder.
        _wr_ran = bool(getattr(cfg.fix, "witness_replay", False))
        _surv_ids = {id(m) for m in muts}
        fx.witness_admits = ((lambda o, _s=_surv_ids: id(o) in _s)
                             if _wr_ran else None)
        # --- SELECTOR (conversion autopsy §3) --------------------------------
        # "binary": N per-candidate yes/no probes, ranked in code, top confirmed
        # by the model. "letter": the historical A-H menu, byte-identical.
        #
        # KNOWN AND ACCEPTED LOSS: the bounded info-request wrapper (`_mut_rctx`)
        # is a LETTER-menu mechanism — it offers `REQUEST(SHOW_SOURCE, B)`, which
        # needs a lettered option list to point at. Under the binary selector
        # each probe sees exactly one candidate, so there is nothing to name and
        # the wrapper does not run. Measured worth of what is given up: on the
        # half-A bench the request channel converted zero mutation menus, while
        # the A-lock it shares the menu with cost every one of them.
        #
        # The probes deliberately carry `fix_task`, NOT `mut_task`: the staged
        # reasoning block argues for a particular LINE, and pasting that argument
        # into every independent probe would re-import the very correlation the
        # probes exist to remove. The CONFIRM step gets `mut_task` — by then the
        # ranking is fixed and all available evidence should be in front of the
        # one call that can veto it.
        _mut_selector = str(getattr(cfg.fix, "menu_selector", "letter") or "letter")
        try:
            if _mut_selector == "binary":
                # RENDER, IDENTICALLY FOR EVERY CANDIDATE (keep the
                # formatting constant across candidates so the score is not
                # measuring layout). Truncation is NOT silent — an elided render
                # is marked, so a probe answered on a half-shown edit can never
                # be mistaken for one answered on the whole thing.
                _CAP = 200

                def _clip(s):
                    return _truncate_code_change(s, _CAP)

                def _cand_render(opt):
                    _li, _old, _new = opt
                    return (_clip(str(_old).strip()),
                            _clip(" / ".join(l.strip() for l in
                                             str(_new).strip().split("\n"))))

                def _probe_q(opt):
                    _o, _n = _cand_render(opt)
                    return CAND_PROBE_Q.format(task=fix_task, fn=fnpick,
                                               old=_o, new=_n)

                def _confirm_q(opt):
                    _o, _n = _cand_render(opt)
                    return CAND_CONFIRM_Q.format(task=mut_task, fn=fnpick,
                                                 old=_o, new=_n)

                mpick = fx.menu_pick_binary(
                    opts, displays, "mutation", _probe_q, _confirm_q,
                    is_candidate=lambda o: isinstance(o, tuple) and o[0] != "none",
                    probe_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"fixes_bug": "<yes|no>"}.'),
                    confirm_continuation=(
                        'You have used your thinking. Reply now with only: '
                        '{"still_broken": "<yes|no>"}.'))
            else:
                mpick = fx.menu_pick(_mut_prompt, opts, displays, "mutation",
                                     rctx=_mut_rctx)
        finally:
            fx.witness_admits = None
        # --- MICRO-CALL SITE 1: skeptic.symptom_line_link -------------------
        # ARMED 2026-08-29 with a real transport (the three-line closure over
        # fixer._call that detreason/microcall.py:TRANSPORT_CONTRACT specifies).
        # It fires ONLY when the mutation menu's own top-2 margin came back
        # thinner than kernel.margin_tau — i.e. precisely the fragile menus the
        # site exists to help, never the confident ones.
        #
        # It is an EXTRA EVIDENCE SOURCE, not an authority: a VERIFIED answer
        # suffixes " | symptom-link" onto the matching option and the menu is
        # asked once more. An unverified answer does nothing at all — the layer
        # guarantees `verified=False` means the model's atom never left the
        # module, and the deterministic fallback it substitutes is a ranking we
        # already have. Append-only again: membership and order do not move.
        _mm = fx.margins[-1] if fx.margins else {}
        # SELECTOR TELEMETRY into the certificate itself, not only into
        # `margins`: the probe count, the per-candidate scores, the candidate
        # ordering used as tie-break, and the confirm outcome are what the next
        # calibration pass reads.
        if _mut_selector == "binary" and isinstance(_mm, dict) \
                and _mm.get("menu") == "mutation" and _mm.get("binary_probe"):
            cert["stages"].append({"mutation_selector": _mm["binary_probe"]})
        # The symptom-link re-ask re-asks the LETTER menu, which is the A-lock
        # this build exists to remove — and `mpick2` would OVERRIDE the binary
        # selection outright. It is therefore letter-path only. `_thin` also
        # means something different on each path (a letter top-2 logprob margin
        # vs a signed probe-score gap), so comparing it to `margin_tau` under
        # binary would be a scale error on top of the override.
        _mm_letter = _mm if _mut_selector != "binary" else {}
        _thin = (isinstance(_mm_letter, dict) and _mm_letter.get("menu") == "mutation"
                 and isinstance(_mm_letter.get("margin"), (int, float))
                 and _mm_letter["margin"] < cfg.kernel.margin_tau)
        if (_thin and getattr(cfg.fix, "microcall_line_link", False)
                and len(muts) >= 2):
            try:
                from detreason import microcall as _mc

                # LENIENT ADAPTER (rung-4000 autopsy, ranked fix #10). MEASURED:
                # the site fired twice and BOTH firings came back
                # `{"verified": false, "reject_reason": "not_json"}` — a 0% parse
                # rate, i.e. we paid for two calls and the layer never saw an
                # atom. microcall.parse_answer is deliberately strict and stays
                # strict: the leniency belongs on OUR side of the transport, the
                # same place the menus put it. So the adapter normalizes the
                # model's own answer into the contract shape — first by reading
                # an offered id out of prose, then, failing that, by re-asking
                # once for the bare id with a budget too small to write a
                # paragraph, exactly as `_reask_one_letter` does for menus. It
                # can only ever produce an id THIS call offered, so nothing
                # hallucinated can survive the adapter, and a still-unreadable
                # answer is passed through untouched to reject honestly.
                _first = {}
                for _li, _old, _n in muts:
                    _first.setdefault(_li, _old)
                _cands = [(f"L{i + 1}", k, str(v).strip())
                          for i, (k, v) in enumerate(_first.items())]
                _ids = [c[0] for c in _cands]

                def _mc_normalize(text):
                    return _coerce_microcall_answer(text, _ids)

                def _mc_transport(prompt):          # TRANSPORT_CONTRACT
                    text, raw = _call(prompt, seed=cfg.seed, url=cfg.url,
                    max_tokens=budget_for_stage(
                        cfg, "microcall", prompt), n_probs=2,
                    enable_thinking=stage_thinking(cfg, "microcall"),
                    label="microcall:symptom_line_link",
                    continuation=CONTINUATION_MICROCALL_Q)
                    fixed = _mc_normalize(text)
                    if fixed is None and (text or "").strip():
                        rtext, _rraw = _call(
                            MICROCALL_REASK_Q.format(
                                prose=_prompt_json(text or "", 600),
                                ids="/".join(_ids)),
                            seed=cfg.seed, url=cfg.url,
                            max_tokens=budget_for_stage(
                                cfg, "reask", MICROCALL_REASK_Q.format(
                                    prose=_prompt_json(text or "", 600),
                                    ids="/".join(_ids))),
                            enable_thinking=stage_thinking(cfg, "reask"),
                            label="microcall:symptom_line_link:reask",
                            continuation=CONTINUATION_MICROCALL_Q)
                        fixed = _mc_normalize(rtext)
                        cert.setdefault("microcall_reasks", []).append(
                            {"site": "symptom_line_link",
                             "recovered": fixed is not None})
                    return (fixed if fixed is not None else text), raw

                if len(_cands) >= 2:
                    _spec = _mc.symptom_line_link_spec(task_text, _cands)
                    _out = _mc.execute(_spec, _mc_transport,
                                       correlation_id=f"fixer:{fpick}:{fnpick}")
                    _linked = ({c[1] for c in _cands if c[0] == _out["answer"]}
                               if _out.get("verified") else set())
                    cert["stages"].append({"microcall_line_link": {
                        "site_id": _out.get("site_id"),
                        "answer": _out.get("answer"),
                        "verified": bool(_out.get("verified")),
                        "reject_reason": _out.get("reject_reason") or "",
                        "menu_margin": _mm.get("margin"),
                        "linked_line": (sorted(_linked)[0] + 1) if _linked else None,
                    }})
                    if _linked:
                        displays2 = [
                            d + (" | symptom-link" if m[0] in _linked else "")
                            for d, m in zip(menu_displays, muts)]
                        # the tail must mirror `opts` exactly — `none` may have
                        # been withdrawn above, and a displays/opts length
                        # mismatch is an IndexError inside the menu renderer.
                        displays2 = displays2 + displays[len(menu_displays):]
                        # the re-ask is the SAME candidate set, so it carries the
                        # same witness admission (autopsy fix #1a)
                        fx.witness_admits = ((lambda o, _s=_surv_ids: id(o) in _s)
                                             if _wr_ran else None)
                        try:
                            mpick2 = fx.menu_pick(_mut_prompt, opts, displays2,
                                                  "mutation_symptom_link",
                                                  rctx=_mut_rctx)
                        finally:
                            fx.witness_admits = None
                        if mpick2 is not None:
                            mpick = mpick2
            except Exception as _e:      # micro-call is additive, never fatal
                cert.setdefault("warnings", []).append(
                    f"microcall line link unavailable ({type(_e).__name__})")
        if mpick is not None and mpick[0] == "none":
            cert["stages"].append({"mutation_menu": "declined"})
            mpick = None
        if mpick is not None:
            li, _, newline = mpick
            c2 = _edit_sets.apply_candidate(lines, mpick)          # SEAM
            try:
                if c2 is None:           # anchor drifted: fail closed
                    raise SyntaxError("candidate could not be anchored")
                compile("\n".join(c2), fpick, "exec")
                new_src = "\n".join(c2) + "\n"
                cert["stages"].append({"fix": "mutation_menu",
                                       "function": fnpick,
                                       "line": li + 1,
                                       "line_span": _cert_line_span(mpick,
                                                                     li + 1),
                                       "change": _cert_change(mpick[1], newline),
                                       "change_old": mpick[1],
                                       "change_new": newline})
            except SyntaxError:
                pass
    # --- rung 2: line menu + single-line region emission ---
    if (new_src is None and cfg.fix.line_fix_fallback
            and str(getattr(cfg.fix, "line_stage_path", "off") or "off") != "off"):
        # LINE_STAGE PATH (wired 2026-08-29). The anchor universe replaces the
        # single model-chosen line; `off` below is the untouched legacy rung.
        new_src, _lscert = _line_stage_rung(
            call=_call, cfg=cfg, cert=cert, task=fix_task, fpick=fpick,
            fnpick=fnpick, lines=lines, lo=lo, hi=hi, span=span,
            fix_retries=fix_retries, repair_task=task_text)
    elif new_src is None and cfg.fix.line_fix_fallback:
        opts = [(lo + i, l) for i, l in enumerate(span) if l.strip()][:26]
        displays = [_truncate_for_prompt(l.strip(), 90) for _, l in opts]
        lpick = fx.menu_pick(
            lambda menu: LINE_MENU_Q.format(task=fix_task, fn=fnpick, menu=menu),
            opts, displays, "line",
            rctx=RequestContext(
                "line", files,
                lambda o, _p=fpick, _f=fnpick: {"path": _p, "func": _f,
                                                "line": int(o[0])},
                cfg, path=fpick))
        if lpick is not None:
            li, old = lpick
            # run-local: cfg default unless the attention allocator raised it
            for attempt in range(max(1, fix_retries)):
                emission_prompt = _build_line_emission_prompt(
                    task=task_text, fn=fnpick, lines=lines,
                    line_no=li + 1, lo=lo, hi=hi)
                text, _ = _call(
                    emission_prompt, seed=cfg.seed + attempt, url=cfg.url,
                    max_tokens=budget_for_stage(
                        cfg, "line_emission", emission_prompt),
                    enable_thinking=stage_thinking(cfg, "line_emission"),
                    label="line_emission",
                    continuation=CONTINUATION_LINE_EMISSION_Q)
                out = text or ""
                cand_line = _extract_replacement_line(out)
                if not cand_line or degenerate(cand_line) or cand_line.strip() == old.strip():
                    continue
                c2 = list(lines)
                indent = len(old) - len(old.lstrip())
                c2[li] = " " * indent + cand_line
                try:
                    compile("\n".join(c2), fpick, "exec")
                except SyntaxError:
                    continue
                # STRUCTURAL GUARD (rung-4000 autopsy, ranked fix #9). Free-form
                # emission with a compile check alone is fail-plausible (#181c):
                # on `log_handler` it wrote
                #     if levelno >= logging.CRITICAL:  ->  def _map_level(...) -> str:
                # — the function's OWN def line, copied over a condition — and the
                # run still reported FIX_PROPOSED. Both halves of that emission are
                # detectable without a model: the replacement DUPLICATES a line
                # already in the function, and it changes the line's top-level AST
                # node type (If -> FunctionDef). Neither is ever a repair.
                bad = _emission_defect(span, old, cand_line)
                if bad:
                    cert["stages"].append({"line_emission_rejected": {
                        "line": li + 1, "reason": bad,
                        "candidate": cand_line[:120]}})
                    continue
                new_src = "\n".join(c2) + "\n"
                indent = len(old) - len(old.lstrip())
                cert["stages"].append({
                    "fix": "line_emission", "function": fnpick,
                    "line": li + 1, "line_span": [li + 1, li + 1],
                    "change": f"{old.strip()} -> {cand_line}",
                    "change_old": old,
                    "change_new": " " * indent + cand_line,
                })
                break
            # EXHAUSTED WITHOUT A CLEAN EMISSION -> the rung produced nothing.
            # A rejected emission must fall through to "ladder exhausted", never
            # to a FIX_PROPOSED, which is the whole point of the guard.
            if new_src is None:
                cert["stages"].append({"line_emission": "no admissible emission"})
    if new_src is None:
        cert["reason"] = "fix ladder exhausted (mutation menu + line emission)"
        return cert
    # ---- PATCH RELEVANCE GATE (post-authoring, pre-acceptance) --------------
    # This is deliberately weaker than behavioural verification: it only proves
    # that the emitted delta is non-empty and touches syntax belonging to the
    # diagnosed class (or shares a symptom term). A syntactically valid edit on
    # an unrelated function must not become a proposed repair merely because it
    # compiles; an unclassifiable family remains UNKNOWN and continues to the
    # existing evidence/regression gates.
    if getattr(cfg.gates, "patch_relevance", True):
        _change_stage = next(
            (
                stage
                for stage in reversed(cert["stages"])
                if isinstance(stage, dict)
                and isinstance(stage.get("change_old"), str)
                and isinstance(stage.get("change_new"), str)
            ),
            None,
        )
        if _change_stage is None:
            _patch_relevance = {
                "decision": "unknown",
                "reason": "no_raw_change_pair",
            }
        else:
            _patch_relevance = _function_relevance.patch_gate(
                _change_stage["change_old"],
                _change_stage["change_new"],
                cls=cls,
                task_text=task_text,
                symptom_terms=set(symptom_vocabulary(task_text)),
            ).as_dict()
        cert["stages"].append({"patch_relevance": _patch_relevance})
        cert["gates"].append(
            "patch_relevance:%s" % (
                "pass"
                if _patch_relevance.get("decision") == "allow"
                else "FAIL"
                if _patch_relevance.get("decision") == "reject"
                else "unknown"
            )
        )
        if _patch_relevance.get("decision") == "reject":
            cert["reason"] = (
                "fix ladder: authored patch has no deterministic relationship "
                "to the diagnosed behaviour"
            )
            return cert
    # compile gate is unconditional — a non-compiling diff is never a fix
    cert["gates"].append("compile:pass")
    # EVIDENCE-LINKAGE gate (smoke-1: a confident wrong diff shipped on compile
    # alone). The changed line must share a symbol with the task text, or the
    # verdict self-downgrades — never confidently unlinked.
    changed = next((s for s in cert["stages"] if "change" in s), None)
    if cfg.gates.evidence_linkage:
        linked = False
        if changed:
            # DIGITS ARE EVIDENCE (autopsy fix #2, second half). The token regex
            # was `[a-z_]{3,}`, which excludes every numeral by construction. The
            # measured cost: `book_planning_service::_adult_audience_block`'s gold
            # change is `if target_age > 18:` -> `< 18:` and the ticket says
            # "Returns an adult-audience directive for learners 18+". `18` is the
            # ONLY token the two sides share, and the character class threw it
            # away. Widened to `[a-z_0-9]{2,}` on BOTH sides so a numeric literal
            # can link. `_LINKAGE_STOP` then removes the Python keywords the
            # widening newly admits: `if`/`or`/`and`/`in` appear in essentially
            # every mutation candidate this harness emits (they ARE what the
            # families flip), so admitting them as evidence would make the gate
            # pass on grammar alone — a gate that always passes is not a gate.
            cw = (words(changed["change"])
                  | {t for t in re.findall(r"[a-z_0-9]{2,}",
                                           changed["change"].lower())
                     if t not in _LINKAGE_STOP})
            tw = (words(task_text) | attr_mentions
                  | {t.lower() for t in code_like}
                  | set(re.findall(r"[a-z_0-9]{2,}", (task_text or "").lower())))
            symptom_anchor_linked = (sym_anchor is not None
                                     and sym_anchor == (fpick, fnpick))
            # NEW CHANNELS: a correct fix inside a function whose
            # DOCSTRING the task quotes verbatim, and an exact-symbol anchor that
            # matched the picked file/function, are both real provenance — they
            # used to self-downgrade to LOW_CONFIDENCE_PROPOSAL and never apply.
            # SCOPED to the CLOSED rungs. The line-emission rung is free-form
            # model output: letting a quoted docstring alone gate-pass an
            # arbitrary emitted line is exactly the fail-plausible pattern the
            # crucible flagged P0. Mutation candidates are template-assembled and
            # compile-checked, so provenance suffices there.
            #
            # WIDENED to witness_replay (autopsy fix #2). MEASURED: on
            # `book_planning_service::_adult_audience_block` witness replay
            # reduced 3 candidates to 1, that 1 was the BYTE-EXACT gold patch, and
            # this gate downgraded it to LOW_CONFIDENCE_PROPOSAL — which never
            # applies — solely because the rescue channel was spelled
            # `== "mutation_menu"`. A witness_replay fix is a mutation candidate
            # that ALSO compiled and ALSO flipped rows on derived boundary
            # witnesses: strictly stronger provenance than a menu pick, and it
            # was the one rung the channel excluded. #197 — the measured half
            # outranks the soft heuristic.
            docstring_linked = (changed.get("fix") in _DOCSTRING_LINKED_RUNGS
                                and _task_quotes_docstring(task_text, node))
            sym_anchor_linked = (sym_anchor is not None
                                 and sym_anchor[0] == fpick
                                 and sym_anchor[1] == fnpick)
            # EXECUTED EVIDENCE (2026-08-29 final-four autopsy, task #1).
            # MEASURED: on `utils/image_processing.py::validate_file_count`
            # spec_exec DECIDED the BYTE-EXACT gold patch (`file_count > 1` ->
            # `< 1`), eliminating three siblings with a synthesized
            # discriminating test that ran them, and this gate withheld it —
            # `fixed=0` on a correct fix. Every existing channel missed by
            # construction: the changed line's only tokens are `file_count`
            # while the ticket says "files", and `_task_quotes_docstring`
            # requires a >=60-char chunk while the synthetic ticket quotes only
            # the 38-char docstring SUMMARY (the Args:/Raises: sections push the
            # full docstring past what the ticket ever contains).
            #
            # An EXECUTED discriminating test is the strongest provenance this
            # harness can produce: the candidate was not merely consistent with
            # the ticket's vocabulary, it was RUN against synthesized inputs and
            # separated from its siblings by observed behaviour. #197 — the
            # measured half outranks the lexical heuristic.
            #
            # Deliberately NARROW, so the gate keeps its teeth: a spec_exec
            # result that merely survived (`suspicious_unique`, or affirmed only
            # by the direction heuristic) is NOT linkage and still self-
            # downgrades. Only an affirmative `discriminating_test` counts, and
            # only on the closed rungs.
            ev = changed.get("evidence") or {}
            executed_linked = (
                changed.get("fix") in _DOCSTRING_LINKED_RUNGS
                and isinstance(ev, dict)
                and "discriminating_test" in (ev.get("affirmative") or ())
                and ev.get("testgen_verdict") == "SELECTED")
            # IDF-WEIGHTED SYMPTOM<->EDITED-IDENTIFIER CHANNEL.
            # ADDITIVE: one more disjunct, LAST, so it can only be REACHED when
            # every stronger channel already declined. It therefore cannot mask
            # a stronger channel's opinion, cannot change what any existing
            # channel decides, and is reported under its own name.
            #
            # SCOPED to the closed rungs for exactly the reason the docstring
            # channel is: the line-emission rung is free-form model output, and
            # letting a lexical overlap gate-pass an arbitrary emitted line is
            # the fail-plausible pattern the crucible flagged P0.
            idf_linked, idf_receipt = False, None
            if (not (bool(cw & tw) or bool(anchors and fpick == anchors[0][0])
                     or symptom_anchor_linked or docstring_linked
                     or sym_anchor_linked or executed_linked)
                    and changed.get("fix") in _DOCSTRING_LINKED_RUNGS
                    and getattr(cfg.gates, "evidence_idf_channel", True)):
                try:
                    _old, _new = _changed_line_pair(changed)
                    idf_linked, idf_receipt = symptom_identifier_link(
                        task_text, _old, _new, build_idf(files))
                except Exception as _e:      # law 1: fail CLOSED to no link
                    idf_receipt = {"channel": "symptom_identifier_lexical_v1",
                                   "result": "error:%s" % type(_e).__name__}
                if idf_receipt is not None:
                    cert["evidence_idf_channel"] = idf_receipt
            linked = (bool(cw & tw)
                      or bool(anchors and fpick == anchors[0][0])
                      or symptom_anchor_linked
                      or docstring_linked
                      or sym_anchor_linked
                      or executed_linked
                      or idf_linked)
            if linked:
                cert["evidence_linkage_by"] = (
                    "changed_line_tokens" if bool(cw & tw)
                    else "file_anchor" if bool(anchors and fpick == anchors[0][0])
                    else "sym_anchor" if (symptom_anchor_linked or sym_anchor_linked)
                    else "function_docstring" if docstring_linked
                    else "executed_evidence" if executed_linked
                    else "symptom_identifier_lexical_v1")
        cert["gates"].append(f"evidence_linkage:{'pass' if linked else 'FAIL'}")
        verdict = "FIX_PROPOSED" if linked else "LOW_CONFIDENCE_PROPOSAL"
    else:
        cert["gates"].append("evidence_linkage:off")
        verdict = "UNGATED_PROPOSAL"
    diff = "".join(difflib.unified_diff(
        src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        fromfile=f"a/{fpick}", tofile=f"b/{fpick}"))
    cert["verdict"] = verdict
    cert["diff"] = diff
    # ---- CERTIFICATE INTEGRITY: the run is checkable from the cert alone ----
    cert["file"] = fpick
    cert["repo_root"] = str(pathlib.Path(root).resolve())
    cert["source_sha256"] = src_sha
    cert["patched_sha256"] = _sha256_text(new_src)
    cert["diff_sha256"] = _sha256_text(diff)
    if fx.parse_fallback:
        cert["parse_fallback"] = True
    cert["wall_s"] = round(time.time() - t0, 1)
    if apply:
        # Apply ONLY on the fully-gated success verdict. A low-confidence or
        # ungated proposal is a proposal — it never touches the working tree.
        if verdict != "FIX_PROPOSED":
            cert["applied"] = False
            cert["apply_reason"] = (
                f"not applied: verdict {verdict} is not the fully-gated "
                f"success verdict FIX_PROPOSED")
            return cert
        try:
            root_r, target = _resolve_inside(root, fpick)
        except ValueError as e:
            cert["applied"] = False
            cert["apply_reason"] = str(e)
            return cert
        # TOCTOU guard: re-read and re-verify the pre-image hash immediately
        # before writing. Anything that moved under us aborts the apply.
        try:
            current = target.read_text()
        except Exception as e:
            cert["applied"] = False
            cert["apply_reason"] = f"apply: source unreadable at write time ({type(e).__name__})"
            return cert
        if _sha256_text(current) != src_sha:
            cert["verdict"] = "STALE_SOURCE"
            cert["applied"] = False
            cert["apply_reason"] = (
                "apply: source changed between read and write (TOCTOU); "
                "the diff no longer describes the file on disk")
            cert["observed_sha256"] = _sha256_text(current)
            return cert
        _atomic_write(target, new_src)
        cert["applied"] = True
    return cert


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('usage: python3 fixer.py <repo_root> "task text" [--apply]')
        sys.exit(1)
    cert = run(sys.argv[1], sys.argv[2], apply="--apply" in sys.argv)
    # The FULL certificate is emitted — diff and config included. Stripping
    # fields for readability made the printed cert unverifiable.
    print(json.dumps(cert, indent=2))
    d = cert.get("diff")
    if d:
        print("\n" + d)
