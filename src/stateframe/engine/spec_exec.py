"""SPECULATIVE EXECUTION AS ARBITER — the mutation menu's model call, replaced.

THE IDEA
--------
The mutation menu asks a model "which of these K candidates is right?". That call
is a position readout (E49): the model is reading the menu, not the code. When the
candidate set is CLOSED — every candidate the enumerators can produce, no sampling —
we do not have to ask. We can APPLY all K of them in isolated namespaces and run
every deterministic check we own. Survival picks.

    arbitrate(fn_source, candidates, symptom_class, span_lines) -> SpecExecResult

K is always `len(candidates)`. There is no top-K sampling: a candidate omitted from
the pool can never be recovered downstream, so omitting one is the one mistake this
arbiter is not allowed to make.

THE CHECK BATTERY (per candidate, all four, none optional)
----------------------------------------------------------
1. `compile`   — the candidate is anchored into the span (intent_oracle._apply,
                 which anchors on the OLD LINE's text because mutation indices are
                 absolute file offsets while the span is span-relative) and the
                 result is parsed and compiled. A candidate that cannot be built is
                 eliminated: it is not a fix, it is a syntax error.
2. `witness`   — witness_replay._judge, called READ-ONLY, exactly as the live
                 filter calls it: boundary-witness truth tables for boolean
                 candidates, the structural contact-family witness for guard
                 insertions / await wraps / kwarg passthroughs / renames. Its
                 contract holds here unchanged: UNSUPPORTED ALWAYS SURVIVES.
3. `testgen`   — ONE call to test_gen_oracle.synthesize_discriminating_test over
                 the whole menu. That is the arbiter's teeth: it is the only check
                 that runs the candidates against synthesized boundary inputs and
                 compares behaviour, and it is the 7-gold, zero-elimination
                 selector measured on this bench. SELECTED narrows to one;
                 AMBIGUOUS narrows to its survivors; NO_SIGNAL is NOT APPLICABLE
                 (it abstains, it does not eliminate).
4. `direction` — the symptom's own words. A `cmp_flip` symptom says the comparison
                 runs the wrong way; a candidate that reverses it is consistent
                 with the symptom (witness_replay.reverses_direction) and one that
                 rewrites an operand is not. Where the symptom names no direction
                 the check is NOT APPLICABLE and stays silent.

An inapplicable check is never a pass that counts and never a failure. Every check
fails OPEN: an engine error is recorded and the candidate keeps standing.

THE VERDICT
-----------
    DECIDED(candidate, evidence) — exactly one full survivor, AND it carries at
        least one AFFIRMATIVE piece of evidence, AND nothing contradicts the
        symptom direction. The model is never called.
    SHORTLIST(survivors)         — more than one full survivor. The menu runs, on
        the shortlist only.
    NO_SIGNAL                    — zero full survivors. The menu runs UNCHANGED
        (every candidate restored) and the abstention is recorded.

THE CLOSED-BUT-GOLD-ABSENT GUARD
--------------------------------
A closed candidate set does not guarantee the gold is IN it. When the gold is
absent, a unique survivor is not "the answer" — it is the least-bad wrong answer,
and auto-applying it is exactly the confident-wrong-answer failure this kernel
exists to avoid. So survival alone never DECIDES. DECIDED additionally requires
AFFIRMATIVE evidence — a discriminating test that cited this candidate, a reversal
that matches the direction the symptom names, or a structural contact witness that
fired — and requires that the direction check is not NEGATIVE. A unique survivor
that clears the vetoes but earns no affirmative evidence is downgraded to
SHORTLIST-of-one, so the menu still runs and the "none of these" option is still
reachable downstream. We would rather spend one model call than land a wrong patch.

WIRING CONTRACT (for fixer.py's mutation stage — NOT wired in this change)
--------------------------------------------------------------------------
fixer owns `fixer.py`; this module is standalone and imports nothing from it. The
mutation stage wires in like this, in ONE place, immediately before the menu is
rendered and after the candidate list is final:

    import spec_exec
    sx = spec_exec.arbitrate(fn_source, candidates, symptom_class, span_lines)
    if sx.verdict == spec_exec.DECIDED:
        chosen = sx.selected                 # skip the menu; no model call
    else:
        menu_candidates = sx.menu_candidates # SHORTLIST -> filtered; NO_SIGNAL -> unchanged
        chosen = <existing menu call>(menu_candidates)
    cert.setdefault("stages", []).append(sx.as_cert_stage())

Invariants the caller must honour:
  * `candidates` is the CLOSED menu, in the fixer's own deterministic order.
    `sx.menu_candidates` preserves that relative order, always.
  * `symptom_class` is the class string the harness already infers
    (`witness_replay.infer_symptom_class(task_text)`); None is legal and simply
    disables the direction check.
  * `span_lines` is the span as a line list. `fn_source` should be
    "\n".join(span_lines); if they disagree, span_lines wins for anchoring.
  * The "none of these" menu option is preserved for every non-DECIDED verdict,
    including SHORTLIST-of-one. Never auto-apply a shortlist.
  * On DECIDED the caller still runs its own apply-safety gate. This arbiter picks
    a candidate; it does not authorise a write.

LEDGER NOTE (append-only): filtering the menu is not a ledger deletion. The
certificate stage records EVERY candidate — survivors, eliminated (with the exact
check that killed each one), and the evidence behind the verdict — so the record
grows. A decider is allowed to shrink the QUESTION; it is not allowed to shrink the
record of what was asked.

Determinism: no randomness, no clock, no network, no shared namespace between
candidates, and every output list is built in candidate order. Two calls on equal
input produce equal results.
"""

import ast
import re
import textwrap
from dataclasses import dataclass, field

import edit_sets as _es
import intent_oracle as io
import test_gen_oracle as tgo
import witness_replay as wr

# ------------------------------------------------------------------ verdicts
DECIDED = "DECIDED"
SHORTLIST = "SHORTLIST"
NO_SIGNAL = "NO_SIGNAL"

#: The four checks, in the order they are run and reported.
#:
#: `set_consistency` is a FIFTH check that only speaks about coordinated
#: `EditSet` candidates (n/a for every single-line candidate, so the existing
#: four-check behaviour is byte-identical for every existing family). It is the
#: cross-line half of the verify-multiline design: the atoms are arbitrated as
#: ONE set, and a set that only holds together because a coordinated atom was
#: dropped is rejected rather than shortlisted.
CHECKS = ("compile", "witness", "testgen", "direction", "set_consistency")

#: Narrowers whose verdict is POSITIVE about the survivor, and may therefore
#: satisfy the closed-but-gold-absent guard. `prose_line` is excluded on
#: purpose: it removes a non-candidate, which is not a statement about the one
#: left standing. See the affirmative block in `_arbitrate`.
_AFFIRMING_NARROWERS = frozenset(("sibling_consistency", "name_subject"))

# Per-check status values.
PASS = "pass"
FAIL = "fail"
NA = "n/a"          # the check does not speak about this candidate
ERROR = "error"     # the check itself broke; fails open, counts as NA


@dataclass
class SpecExecResult:
    """The arbitration record. Complete enough to be the certificate stage."""

    verdict: str = NO_SIGNAL
    selected: object = None              # the DECIDED candidate, else None
    survivors: list = field(default_factory=list)      # full survivors, in order
    eliminated: list = field(default_factory=list)     # [{index, line, failed, reason}]
    evidence: list = field(default_factory=list)       # one dict per candidate
    affirmative: list = field(default_factory=list)    # affirmative sources for the pick
    cases_cited: list = field(default_factory=list)    # discriminating-test labels
    reason: str = ""
    candidates: list = field(default_factory=list)     # the closed menu, as given
    testgen_verdict: str = ""
    suspicious_unique: bool = False      # unique survivor, no affirmative evidence
    narrowed_by: str = ""                # in-function narrower that broke a tie
    # True when survivors == candidates: the battery ran and eliminated NOTHING. Distinct
    # from an ordinary SHORTLIST, which did real narrowing and still could not close.
    # See FINDINGS E246 — this is the dominant reason a run that already located the bug
    # abstains. Reporting only; it never changes a verdict or a survivor set.
    zero_information: bool = False

    @property
    def menu_candidates(self):
        """What the mutation menu should be run over.

        DECIDED   -> [] (the menu does not run)
        SHORTLIST -> the survivors, original relative order preserved
        NO_SIGNAL -> the menu UNCHANGED
        """
        if self.verdict == DECIDED:
            return []
        if self.verdict == SHORTLIST:
            return list(self.survivors)
        return list(self.candidates)

    @property
    def counts(self):
        return {"candidates": len(self.candidates),
                "survivors": len(self.survivors),
                "eliminated": len(self.eliminated)}

    def as_cert_stage(self):
        """The append-only certificate record for this arbitration."""
        return {
            "stage": "spec_exec",
            "verdict": self.verdict,
            "selected": self.selected,
            "survivors": list(self.survivors),
            "eliminated": [dict(e) for e in self.eliminated],
            "evidence": [dict(e) for e in self.evidence],
            "affirmative": list(self.affirmative),
            "cases_cited": list(self.cases_cited),
            "testgen_verdict": self.testgen_verdict,
            "suspicious_unique": self.suspicious_unique,
            "narrowed_by": self.narrowed_by,
            "counts": self.counts,
            "reason": self.reason,
        }


# ============================================================ 1. THE CHECKS
def _applied_source(span_lines, mut):
    """The candidate's function source, or None. Edit-set aware.

    A coordinated `EditSet` carries `line_span`, so it is applied as ONE atomic
    block splice against the unmodified span (`edit_sets.apply_to_span`, which
    re-checks the exact-match anchor and fails closed). Anything else takes
    `intent_oracle._apply` unchanged — the single-line path is byte-identical to
    what it was before this branch existed.
    """
    if getattr(mut, "line_span", None) is not None:
        applied = _es.apply_to_span(span_lines, mut)
        if applied is None:
            return None
        src = "\n".join(applied)
        try:
            ast.parse(textwrap.dedent(src))
        except SyntaxError:
            return None
        return src
    return io._apply(span_lines, mut)


def _check_compile(span_lines, mut):
    """Anchor the candidate into the span and compile it. (status, detail)."""
    try:
        src = _applied_source(span_lines, mut)
    except Exception as exc:                     # law 1: rather learn nothing
        return ERROR, "apply engine error: %s" % type(exc).__name__
    if src is None:
        return FAIL, "cannot be anchored into the span, or does not parse"
    try:
        compile(ast.parse(textwrap.dedent(src)), "<candidate>", "exec")
    except SyntaxError as exc:
        return FAIL, "does not compile: %s" % exc.msg
    except Exception as exc:
        return ERROR, "compile engine error: %s" % type(exc).__name__
    return PASS, "compiles"


def _check_witness(symptom_class, span_lines, mut):
    """witness_replay._judge, read-only. Unsupported always survives."""
    try:
        ev = wr._judge(symptom_class, mut, span_lines)
    except Exception as exc:
        return ERROR, "witness engine error: %s" % type(exc).__name__, {}
    # AN ABSTENTION IS NOT A PASS (FINDINGS E246). `_judge` fails open — its `base` dict is
    # built with survived=True — so any candidate it cannot speak about came back PASS. That
    # made `witness` return PASS on 3,232 of 3,232 candidates: a check that never once failed
    # and never once eliminated anything, while still reading as verification on every
    # evidence row. Rule #7: a gate that cannot fail is worse than no gate.
    #
    # `_judge` already publishes `supported`, distinguishing "I judged this and it survived"
    # from "I have nothing to say about this". Honour it. This CANNOT cause a wrong
    # elimination — NA never eliminates — it only stops an abstention from being counted as
    # evidence of correctness, and makes the real witness coverage visible in the ledger.
    if not ev.get("supported"):
        return NA, ev.get("reason") or "witness has no judgement for this candidate", ev
    status = PASS if ev.get("survived") else FAIL
    return status, ev.get("reason", ""), ev


def _testgen_matrix(fn_source, candidates, symptom_class, span_lines):
    """ONE synthesize_discriminating_test call -> per-candidate status.

    Returns (statuses, details, result). NO_SIGNAL makes every status NA: the
    oracle abstaining is not an elimination.
    """
    n = len(candidates)
    try:
        res = tgo.synthesize_discriminating_test(fn_source, candidates,
                                                 symptom_class, span_lines)
    except Exception as exc:                     # The inner check also fails open.
        return [ERROR] * n, ["testgen engine error: %s" % type(exc).__name__] * n, None
    verdict = getattr(res, "verdict", tgo.NO_SIGNAL)
    if verdict == tgo.NO_SIGNAL:
        # PROPAGATE THE ORACLE'S OWN REASON (FINDINGS E246). This used to substitute a single
        # generic string for every candidate, which destroyed the only evidence that
        # distinguishes "no oracle predicate fired" (a coverage problem -> write predicates)
        # from "engine error: X" (a bug -> fix the engine). testgen is n/a on 90% of
        # candidates and supplies 81% of all eliminations when it does fire, so that 90% is
        # the single highest-leverage population on the bench — and until now the ledger
        # could not tell us which half of it we were looking at.
        why = getattr(res, "reason", None) or "no discriminating signal"
        return [NA] * n, ["the test oracle abstained: %s" % why] * n, res
    if verdict == tgo.SELECTED and res.selected is not None:
        keep = {id(res.selected)}
        why_in = "the discriminating test selected this candidate"
        why_out = "a discriminating test separated this candidate from the winner"
    else:                                        # AMBIGUOUS
        keep = {id(s) for s in (res.survivors or [])}
        why_in = "survives every discriminating test"
        why_out = "killed by a discriminating test"
    statuses, details = [], []
    for cand in candidates:
        if id(cand) in keep:
            statuses.append(PASS)
            details.append(why_in)
        else:
            statuses.append(FAIL)
            details.append(why_out)
    return statuses, details, res


def _check_set_consistency(span_lines, mut):
    """Set-level checks. NA for anything that is not a coordinated `EditSet`.

    Five relational checks, all fail-CLOSED (the whole set is rejected, never
    trimmed to the atoms that passed):

      set_atomicity          every atom applies against the ORIGINAL snapshot,
                             or none does. A partial application is not a
                             candidate — it is not even evidence.
      coupling               every atom shares a symbol with the rest of the
                             set; no orphan rides along on its own score.
      value_threaded         every name the set INTRODUCES is both bound and
                             read in the composed function (no write-only
                             binding, no read with no writer — #223's shape).
      both_branches_updated  when the set edits one arm of an if/else, the
                             sibling arm may not still read the value the set
                             just replaced.
      no_stale_reads         the symbol a retargeting atom replaced is not still
                             read at another site the set left behind.
    """
    if getattr(mut, "line_span", None) is None:
        return NA, "not a coordinated edit-set", {}
    try:
        composed = _es.apply_to_span(span_lines, mut)
        if composed is None:
            return FAIL, "set_atomicity: the set does not apply atomically", {}
        if not _es.coupling_ok(mut.edits):
            return FAIL, "coupling: an atom shares no symbol with the set", {}
        tree = ast.parse(textwrap.dedent("\n".join(composed)))
        base = ast.parse(textwrap.dedent("\n".join(span_lines)))
    except SyntaxError:
        return FAIL, "the composed function does not parse", {}
    except Exception as exc:
        return ERROR, "set engine error: %s" % type(exc).__name__, {}
    detail = {"pattern": mut.pattern, "edits": mut.n_edits,
              "coupling": list(mut.coupling)}
    stores, loads = _name_roles(tree)
    b_stores, _b_loads = _name_roles(base)
    introduced = stores - b_stores
    orphan = sorted(n for n in introduced if n not in loads)
    if orphan:
        return FAIL, ("value_threaded: %s is bound by the set and never read"
                      % ", ".join(orphan)), detail
    stale = sorted(_stale_reads(mut, composed))
    if stale:
        return FAIL, ("no_stale_reads: %s still reads the replaced source"
                      % ", ".join(stale)), detail
    sib = _sibling_arm_gap(mut, composed)
    if sib:
        return FAIL, "both_branches_updated: %s" % sib, detail
    return PASS, ("set of %d coupled edits applies atomically and threads %s"
                  % (mut.n_edits, ", ".join(mut.coupling) or "its symbols")), detail


def _name_roles(tree):
    stores, loads = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (stores if isinstance(n.ctx, ast.Store) else loads).add(n.id)
    return stores, loads


def _stale_reads(mut, composed):
    """Exact source texts a retargeting atom replaced that still appear."""
    body = "\n".join(composed)
    out = set()
    for ed in mut.edits:
        if not ed.role.startswith("retarget") or not ed.old or ed.old == ed.new:
            continue
        # A hoist DELIBERATELY re-introduces the replaced text once, as the
        # binding's value — that occurrence is the fix, not a stale read. Only
        # occurrences no sibling atom put there count against the set.
        reintroduced = sum(1 for o in mut.edits
                           if o is not ed and ed.old in (o.new or ""))
        if body.count(ed.old) > reintroduced:
            out.add(ed.old.strip().splitlines()[0][:60])
    return out


def _sibling_arm_gap(mut, composed):
    """If the set edits inside one arm of an if/else, check the other arm.

    A branch whose sibling still reads a name the set retargeted away is the
    classic one-sided coordinated fix. Silence (no if/else, or nothing
    retargeted) is not a veto.
    """
    replaced = set()
    for ed in mut.edits:
        if ed.role.startswith("retarget"):
            replaced |= {t for t in _es._symbols(ed.old)
                         if t not in _es._symbols(ed.new)}
    if not replaced:
        return ""
    try:
        tree = ast.parse(textwrap.dedent("\n".join(composed)))
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and node.orelse):
            continue
        body_syms = set()
        for st in node.body:
            body_syms |= _es._symbols(ast.dump(st))
        else_syms = set()
        for st in node.orelse:
            else_syms |= _es._symbols(ast.dump(st))
        touched = replaced & body_syms
        if touched and (replaced & else_syms) and not (touched & else_syms):
            return "one arm was retargeted, the sibling arm was not"
    return ""


def _direction(symptom_class, mut):
    """True / False / None for "does this candidate move the way the symptom says".

    None means the symptom names no direction we can read — silence, never a veto.
    Only `cmp_flip` speaks today; direction detection is witness_replay's
    `reverses_direction`, called rather than reimplemented.
    """
    if symptom_class != "cmp_flip":
        return None
    try:
        _idx, old_line, new_line = mut
        old_node, _ = wr.extract_condition(old_line)
        new_node, _ = wr.extract_condition(new_line)
        if old_node is None or new_node is None:
            return None
        return bool(wr.reverses_direction(old_node, new_node))
    except Exception:
        return None


# ------------------------------------------------- 1b. IN-FUNCTION NARROWERS
#
# The four checks above ask "is this candidate consistent with the SYMPTOM".
# These three ask a different and complementary question: "is this candidate
# consistent with the CODE AROUND IT". They exist because the measured failures
# on the 2026-08-29 17/21 board were both cases where every surviving candidate
# passed the whole battery and the arbiter had to hand a genuine tie to the
# mutation menu, which then guessed:
#
#   api/code_submissions.py::_update_lesson_progress — survivors `score > 70`
#       and `score >= 70`. testgen returned NO_SIGNAL because the mutated
#       expression reads an ATTRIBUTE (`lesson_progress.score`), not a
#       parameter, so no probe ladder can vary it. But the SAME function's
#       other branch already writes `passed=(score >= 70)`. The function states
#       its own contract twice; the candidate that makes the two branches agree
#       is the fix, and that is checkable without running anything.
#
#   utils/teacher_personas.py::get_persona_needs_adaptation — four surviving
#       bool_flip candidates, no direction and no discriminating signal. Gold is
#       `not persona_id and persona_id not in PERSONAS` -> `or`: both operands
#       interrogate the SAME subject (`persona_id`), so they are ALTERNATIVE
#       failure modes of one thing and belong under `or`. The decoy the ballot
#       chose, `not has_adhd and not has_autism`, tests two DISTINCT subjects —
#       a De Morgan "neither" guard whose `and` is correct, and which the
#       docstring ("no relevant flags") states outright.
#
# Every narrower is PRUNING, and a prune cannot be recovered downstream, so each
# one fires only when it is UNIQUELY decisive and each is measured against the
# whole task corpus before it ships (test_spec_exec_offline.py). A narrower that
# would empty the survivor set, or that leaves more than one, is a no-op.


def _fn_node(source):
    try:
        tree = ast.parse(textwrap.dedent(source or ""))
    except Exception:
        return None
    return next((n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)


def _is_prose_only_change(mut):
    """True when old->new differs ONLY inside string literals.

    A `bool_flip` enumerator matches " and "/" or " textually, so it happily
    offers prose: a docstring sentence, or — measured 2026-08-29 —
    `logger.warning(f"Report card: course and learner not found...")` in
    `chapter_report_card_service`, which the menu then chose over the real guard
    two lines above it. Rewriting a log message cannot change control flow, so
    such a candidate is not a fix and must never occupy a menu slot.

    Decided structurally: both sides are parsed and every string constant is
    replaced by one placeholder. If the trees then match, nothing but text moved.
    """
    try:
        _idx, old_line, new_line = mut
    except Exception:
        return False
    if old_line is None or new_line is None or old_line == new_line:
        return False

    def _skeleton(line):
        txt = textwrap.dedent(line).strip().rstrip(",")
        for src in (txt, txt.rstrip(":") + ":\n    pass", txt + "\n    pass"):
            try:
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    node.value = "<str>"
                # an f-string's literal chunks live in JoinedStr.values
                if isinstance(node, ast.JoinedStr):
                    node.values = [v for v in node.values
                                   if not isinstance(v, ast.Constant)]
            return ast.dump(tree)
        return None

    a, b = _skeleton(old_line), _skeleton(new_line)
    return a is not None and a == b


def _docstring_line_texts(source):
    """The set of stripped source lines occupied by the function's docstring.

    A mutation landing inside a docstring is prose editing, never a code change:
    it cannot alter behaviour, so it can never be the fix. The enumerators emit
    them because a docstring sentence containing " or " is textually a boolean
    connective (measured: `teacher_personas` candidate 0 was the docstring line
    "Returns '' when the learner has no relevant flags or the id is unknown.").
    """
    node = _fn_node(source)
    if node is None:
        return frozenset()
    doc = ast.get_docstring(node, clean=False)
    if not doc:
        return frozenset()
    return frozenset(ln.strip() for ln in doc.splitlines() if ln.strip())


def _cmp_target_and_literal(line):
    """(target-name, literal, op-class) for `<target> = <x> <op> <literal>`.

    Reads BOTH the assignment form (`lp.passed = lp.score < 70`) and the keyword
    form (`passed=(score >= 70)`) so the two branches of one function can be
    compared. Returns None when the line is not that shape.
    """
    # Both shapes parse as a bare statement once the trailing comma is gone:
    # `lp.passed = lp.score < 70` is an Assign and `passed=(score >= 70)` is
    # also a valid Assign. No rewriting fallback — one that renamed the target
    # to `_` would make two unrelated lines look like siblings.
    txt = textwrap.dedent(line or "").strip().rstrip(",")
    for src in (txt,):
        try:
            mod = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(mod):
            target = None
            value = None
            if isinstance(node, ast.Assign) and node.targets:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.keyword) and node.arg:
                target, value = node.arg, node.value
            if value is None:
                continue
            if not (isinstance(value, ast.Compare) and len(value.ops) == 1
                    and len(value.comparators) == 1):
                continue
            lit = value.comparators[0]
            if not (isinstance(lit, ast.Constant)
                    and isinstance(lit.value, (int, float))
                    and not isinstance(lit.value, bool)):
                continue
            if isinstance(target, str):
                name = target
            elif isinstance(target, ast.Attribute):
                name = target.attr
            elif isinstance(target, ast.Name):
                name = target.id
            else:
                continue
            return name, lit.value, type(value.ops[0]).__name__
    return None


def _sibling_operator(source, mut):
    """The operator the SAME function already uses for the same (target, literal).

    `_update_lesson_progress` writes `passed` twice against the literal 70: once
    on the mutated line and once, correctly, in the else-branch. That second
    write is the function stating its own contract, and it is evidence no probe
    ladder can reach because the mutated expression is an attribute read.
    Returns the operator class name, or None when the function does not repeat
    itself (which is the common case, so the narrower stays silent).
    """
    try:
        _idx, old_line, _new_line = mut
    except Exception:
        return None
    here = _cmp_target_and_literal(old_line)
    if here is None:
        return None
    name, lit, _op = here
    node = _fn_node(source)
    if node is None:
        return None
    old_key = textwrap.dedent(old_line or "").strip()
    ops = set()
    # the ORIGINAL source lines, not ast.unparse: unparse renormalises
    # `passed=(score >= 70),` into `passed=score >= 70`, which then fails to
    # match `old_key` and lets the mutated line vote on ITSELF — measured, and
    # it silenced the narrower on exactly the task it was written for.
    for line in (source or "").splitlines():
        if textwrap.dedent(line).strip() == old_key:
            continue                     # the mutated line never votes on itself
        other = _cmp_target_and_literal(line)
        if other and other[0] == name and other[1] == lit:
            ops.add(other[2])
    # ambiguous self-repetition is silence, not a guess
    return ops.pop() if len(ops) == 1 else None


def _guard_operand_subjects(line):
    """Per-operand name sets for a two-operand boolean guard, else None.

    `not persona_id and persona_id not in PERSONAS` -> [{persona_id},
    {persona_id, PERSONAS}]; `not has_adhd and not has_autism` ->
    [{has_adhd}, {has_autism}].
    """
    txt = textwrap.dedent(line or "").strip()
    if not txt:
        return None
    for src in (txt, txt + "\n    pass", "if " + txt.rstrip(":") + ":\n    pass"):
        try:
            mod = ast.parse(src)
        except Exception:
            continue
        test = next((n.test for n in ast.walk(mod) if isinstance(n, ast.If)), None)
        if test is None:
            test = next((n for n in ast.walk(mod) if isinstance(n, ast.BoolOp)), None)
        if not isinstance(test, ast.BoolOp) or len(test.values) != 2:
            continue
        return [frozenset(x.id for x in ast.walk(v) if isinstance(x, ast.Name))
                for v in test.values]
    return None


_SUBJECT_STOP = frozenset((
    "get", "set", "build", "make", "create", "update", "check", "run", "load",
    "fetch", "compute", "handle", "process", "parse", "format", "render",
    "validate", "ensure", "resolve", "apply", "the", "and", "for", "from",
    "with", "not", "sync", "init", "main", "self", "data", "value", "result",
))


def _name_subjects(fn_name):
    """The SUBJECT tokens of a function name — what it is about, not what it does.

    `get_persona_needs_adaptation` -> {persona, needs, adaptation}. Verbs and
    generic scaffolding are dropped because every function has them and a token
    every candidate shares cannot discriminate anything.
    """
    parts = [p for p in re.split(r"[^a-z0-9]+", (fn_name or "").lower()) if p]
    return frozenset(p for p in parts if len(p) > 3 and p not in _SUBJECT_STOP)


def _guard_mentions_subject(line, subjects):
    """Does this candidate's guard interrogate what the function is ABOUT?

    Measured on `utils/teacher_personas.py::get_persona_needs_adaptation`: three
    candidates survive the whole battery with no direction and no discriminating
    signal, and only the GOLD one (`not persona_id or persona_id not in
    PERSONAS`) names the function's own subject — the decoys guard `has_adhd`,
    `has_autism` and `text`. Returns None when the line is not a readable
    two-operand guard, so an unreadable candidate is never outvoted.
    """
    subs = _guard_operand_subjects(line)
    if subs is None:
        return None
    if not subjects:
        return False
    names = {n.lower() for s in subs for n in s}
    return any(any(sub in n or n in sub for n in names) for sub in subjects)


def _check_direction(symptom_class, mut):
    verdict = _direction(symptom_class, mut)
    if verdict is None:
        return NA, "the symptom names no readable direction", None
    if verdict:
        return PASS, "reverses the comparison direction the symptom names", True
    return FAIL, "does not move the direction the symptom names", False


def _narrow(source, symptom_class, survivors):
    """Break a full-battery TIE using the code around the candidates.

    Runs the narrowers in confidence order and STOPS at the first one that
    leaves exactly one candidate. Each is a prune, so each must be uniquely
    decisive: a narrower that would empty the set, or that leaves the set the
    same size, is discarded and the next one is tried. Returns
    (survivors, narrowed_by) with narrowed_by "" when nothing was decisive.
    """
    docs = _docstring_line_texts(source)

    def _not_prose(muts):
        # UNCONDITIONAL, and not a tie-break: prose is not code, so it is
        # dropped whenever it leaves anything behind, even if several real
        # candidates remain.
        #
        # Two ways a candidate can be prose. It sits in the DOCSTRING — matched
        # on the OLD line, since the candidate's new line has already had its
        # connective swapped and is by construction absent from the docstring.
        # Or its entire diff lives inside STRING LITERALS: measured 2026-08-29,
        # `chapter_report_card_service` offered
        # `logger.warning(f"... course and learner not found ...")` -> `or`, and
        # the menu chose that log message over the real guard two lines above.
        keep = [m for m in muts
                if (textwrap.dedent(_old_line_of(m) or "").strip() not in docs
                    and not _is_prose_only_change(m))]
        return keep or muts

    survivors = _not_prose(survivors)
    if len(survivors) <= 1:
        return survivors, "prose_line" if len(survivors) == 1 else ""

    def _sibling(muts):
        if symptom_class != "cmp_flip":
            return muts
        keep = []
        for m in muts:
            want = _sibling_operator(source, m)
            if want is None:
                return muts          # no self-repetition anywhere: stay silent
            got = _cmp_target_and_literal(_line_of(m))
            if got and got[2] == want:
                keep.append(m)
        return keep

    def _subject(muts):
        if symptom_class != "bool_flip":
            return muts
        node = _fn_node(source)
        subjects = _name_subjects(getattr(node, "name", ""))
        if not subjects:
            return muts
        verdicts = [(m, _guard_mentions_subject(_line_of(m), subjects))
                    for m in muts]
        if any(v is None for _m, v in verdicts):
            return muts              # a candidate we cannot read is not outvoted
        return [m for m, v in verdicts if v]

    for name, rule in (("sibling_consistency", _sibling),
                       ("name_subject", _subject)):
        try:
            kept = rule(survivors)
        except Exception:            # law 1: a narrower that breaks stays silent
            continue
        if len(kept) == 1:
            return list(kept), name
    return survivors, ""


# ========================================================== 2. THE ARBITER
def arbitrate(fn_source, candidates, symptom_class=None, span_lines=None):
    """Apply every candidate, run the full battery, let survival pick.

    See the module docstring for the verdict rules, the gold-absent guard and the
    fixer wiring contract. Never raises: an engine failure is NO_SIGNAL with the
    menu handed back unchanged.
    """
    result = SpecExecResult()
    try:
        return _arbitrate(fn_source, candidates, symptom_class, span_lines, result)
    except Exception as exc:                     # law 1: rather learn nothing
        result.verdict = NO_SIGNAL
        result.reason = "arbiter engine error: %s" % type(exc).__name__
        result.candidates = list(candidates or [])
        result.survivors = []
        return result


def _arbitrate(fn_source, candidates, symptom_class, span_lines, result):
    cands = list(candidates or [])
    result.candidates = cands
    if not cands:
        result.reason = "no candidates to arbitrate"
        return result
    span = list(span_lines) if span_lines else (fn_source or "").splitlines()
    source = fn_source if fn_source is not None else "\n".join(span)
    if len(cands) == 1:
        # A one-candidate menu is not an arbitration. Hand it back untouched so the
        # existing "none of these" path stays reachable.
        result.verdict = NO_SIGNAL
        result.reason = "a single candidate is not a question this arbiter answers"
        result.evidence = [_blank_row(0, cands[0])]
        return result

    tg_status, tg_detail, tg_res = _testgen_matrix(source, cands, symptom_class, span)
    result.testgen_verdict = getattr(tg_res, "verdict", "") or ""
    if tg_res is not None:
        result.cases_cited = list(getattr(tg_res, "cases_cited", []) or [])

    survivors, eliminated, evidence = [], [], []
    for i, mut in enumerate(cands):
        row = _blank_row(i, mut)
        c_status, c_detail = _check_compile(span, mut)
        w_status, w_detail, w_ev = _check_witness(symptom_class, span, mut)
        d_status, d_detail, d_val = _check_direction(symptom_class, mut)
        s_status, s_detail, s_ev = _check_set_consistency(span, mut)
        row["checks"] = {
            "compile": {"status": c_status, "detail": c_detail},
            "witness": {"status": w_status, "detail": w_detail},
            "testgen": {"status": tg_status[i], "detail": tg_detail[i]},
            "direction": {"status": d_status, "detail": d_detail},
            "set_consistency": {"status": s_status, "detail": s_detail},
        }
        if s_ev:
            row["edit_set"] = s_ev
        row["family_witness"] = bool(w_ev.get("family_witness")) if w_ev else False
        row["direction_ok"] = d_val
        failed = [name for name in CHECKS
                  if row["checks"][name]["status"] == FAIL]
        row["failed"] = failed
        row["survived"] = not failed
        evidence.append(row)
        if failed:
            eliminated.append({"index": i, "line": _line_of(mut),
                               "failed": list(failed),
                               "reason": row["checks"][failed[0]]["detail"]})
        else:
            survivors.append(mut)

    result.evidence = evidence
    result.eliminated = eliminated
    result.survivors = survivors

    if not survivors:
        result.verdict = NO_SIGNAL
        result.reason = ("every candidate was eliminated; the menu runs unchanged "
                         "and the abstention is recorded")
        return result
    narrowed_by = None
    if len(survivors) > 1:
        survivors, narrowed_by = _narrow(source, symptom_class, survivors)
        result.survivors = survivors
        result.narrowed_by = narrowed_by
    if len(survivors) > 1:
        result.verdict = SHORTLIST
        # ZERO-INFORMATION FLAG (FINDINGS E246). "N of N survive" and "9 of 14 survive" are
        # both SHORTLIST today, and they are not remotely the same event: the first means the
        # battery ran and learned NOTHING, the second means it did real work and still could
        # not close. Measured, the battery eliminates 6% overall and the no-op case is the
        # single most common reason a run that already LOCATED the bug abstains (214 rows).
        #
        # Recording it as a first-class outcome stops a no-op from reading as considered
        # judgement, and gives callers/healers something to branch on. Deliberately a
        # REPORTING change only: the verdict and survivor set are untouched, so no candidate
        # is eliminated or promoted on the strength of this flag.
        result.zero_information = (len(survivors) == len(cands))
        if result.zero_information:
            result.reason = ("the battery produced ZERO information: all %d candidates "
                             "survive every check" % len(cands))
        else:
            result.reason = ("%d of %d candidates survive the full battery"
                             % (len(survivors), len(cands)))
        return result

    # --- exactly one survivor: the closed-but-gold-absent guard --------------
    only = survivors[0]
    row = next(r for r in evidence if r["survived"] and r["index"] ==
               next(i for i, m in enumerate(cands) if m is only))
    affirmative = []
    if row["checks"]["testgen"]["status"] == PASS \
            and result.testgen_verdict == tgo.SELECTED:
        affirmative.append("discriminating_test")
    if row["direction_ok"] is True:
        affirmative.append("symptom_direction")
    if row["family_witness"]:
        affirmative.append("contact_family_witness")
    if narrowed_by in _AFFIRMING_NARROWERS:
        # A POSITIVE narrower is affirmative evidence in its own right: it did
        # not merely fail to kill this candidate, it read the surrounding code
        # and found that candidate AGREEING with it. Without this the narrowed
        # unique falls through to `suspicious_unique` and is handed straight
        # back to the menu that was guessing in the first place.
        #
        # `prose_line` is deliberately NOT in that set. It is an
        # ELIMINATION — it proves the other candidate was never a code change —
        # and it says nothing whatever about whether the survivor is right. The
        # closed-but-gold-absent guard exists precisely for "last one standing",
        # so a narrower that only removes a non-candidate must not be allowed to
        # satisfy it. It still shrinks the menu; it just cannot end the question.
        affirmative.append(narrowed_by)
    result.affirmative = affirmative

    if affirmative and row["direction_ok"] is not False:
        result.verdict = DECIDED
        result.selected = only
        result.reason = ("unique full survivor, affirmed by %s"
                         % ", ".join(affirmative))
        return result

    result.verdict = SHORTLIST
    result.suspicious_unique = True
    result.reason = ("unique survivor with no affirmative evidence; the candidate "
                     "set is closed but the gold may be absent, so the menu runs "
                     "on this one with the 'none of these' option preserved")
    return result


def _blank_row(index, mut):
    return {"index": index, "line": _line_of(mut),
            "checks": {name: {"status": NA, "detail": "not run"} for name in CHECKS},
            "failed": [], "survived": True,
            "family_witness": False, "direction_ok": None}


def _line_of(mut):
    try:
        return mut[2]
    except Exception:
        return None


def _old_line_of(mut):
    try:
        return mut[1]
    except Exception:
        return None
