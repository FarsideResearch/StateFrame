"""COORDINATED EDIT-SETS — the multi-line family, and the census families.

Why this file exists
--------------------
Census + the v3 real board: 10 of 21 real tasks are ``multi_line_coordinated``.
Every existing family emits ONE ``(line_index, old_line, new_line)`` tuple, so
none of them can express "change the guard AND the write it authorizes". This
module adds the coordinated family without disturbing a single existing one.

Section 4c adds the three CENSUS families on the same substrate — ``import_change``
(3 v3 tasks), ``statement_insert`` (2) and ``exception_handler_change`` (1) — for
the same structural reason: each is a multi-line INSERTION, and a `SpanEdit`
whose ``new`` is ``stmt + "\\n" + old`` is exactly that. They run as a SECOND
tranche with their own budget, after the coordination candidates are already
final, so the coordination menu is byte-identical to what it was.

Their honest ceiling is measured, not asserted: ``probes/census_reachability.py``
reports 0 of 6 gold-reachable, with every miss classified NOVEL-TOKEN or
ASSEMBLY against a 3,519-file corpus. See `CENSUS_PUNTS`.

The substrate (S1 from /tmp/executor_steal_list.md)
---------------------------------------------------
The INTERNAL unit is a **span edit**, not a line index::

    SpanEdit(old="...", new="...", role="...")

``old`` is an exact multi-line substring of the source; ``new`` replaces it and
may be ``""`` (delete is free). This is MC's production ``_code_str_replace``
primitive, and it is deliberately the internal representation even though v1
only *emits* line-tuple-compatible sets: adopting spans later must not require
re-deriving every generator. Its four safety disciplines are ported verbatim in
`apply_span_edits`:

  1. exact byte match, no normalization, no fuzzy fallback -> ``count == 0`` is a
     hard error (anti-anchor-drift by construction);
  2. uniqueness required -> ``count > 1`` is a hard error, never guessed;
  3. size-regression tripwire (a >=100-line source shrunk below 40% is a
     full-file rewrite wearing a str_replace costume);
  4. a parse gate. NOT ported fail-open: the kernel's ``compile()`` is stricter
     and stays stricter (steal-list caution).

The EXTERNAL unit is `EditSet`, a **subclass of the existing 3-tuple**
``(first_line_index, old_block, new_block)`` where the block is the contiguous
run of source lines from the first to the last edited line. Everything
downstream that only unpacks three values (menu display, evidence ranking,
prose filter, ``witness_replay._judge``, the certificate) keeps working
untouched. The ONE thing that changes is application, and only for candidates
that carry ``line_span``:

    lines[lo:hi + 1] = new_block.split("\\n")     # instead of lines[li] = new

`apply_candidate` is that seam, and a plain tuple takes the old path
byte-identically (``getattr(tuple, "line_span", None) is None``).

Fail-closed, always
-------------------
A partial application is never emitted. `apply_span_edits` builds the whole set
against one immutable snapshot and returns ``None`` the moment any single edit
fails to anchor, so a caller can never observe a half-applied set.

Coupling
--------
An edit-set is only generated when its atoms share symbols or dataflow (the
generators below construct that relation directly rather than filtering a
product), and `coupling_ok` re-asserts it on the finished set as a cheap
independent check. There is no Cartesian enumeration anywhere in this file:
every generator walks *from* a seed *along* a coupling edge, which is what makes
the honest caps (8 emitted / 32 explored per function) meaningful rather than a
silent truncation of an exponential space.
"""

from __future__ import annotations

import ast
import re

# ------------------------------------------------------------------ honest caps
#: Emitted per function. Sets are appended AFTER every
#: single-line candidate, so this budget can only ever evict a coordinated
#: candidate — never one from a family that was already measured.
MAX_EDIT_SETS = 8
#: Complete sets a generator may *explore* per function before it stops. Hard.
#: Reaching it is reported, never silently swallowed (see `search_exhausted`).
MAX_EXPLORED_SETS = 32
#: The observed envelope from the v3 board (verify-multiline answer).
MAX_SET_EDITS = 6
#: Bounds on the DIFF, not on the encoding. The contiguous ``old_block`` an
#: `EditSet` carries can legitimately be long (ML-01's two coupled edits sit 103
#: lines apart in the same function); what must stay small is the number of
#: lines the set actually changes and the number of separate hunks it opens.
MAX_CHANGED_LINES = 16
MAX_HUNKS = 4
#: A backstop on the encoding itself: past this the block is a function rewrite
#: wearing an edit-set costume (the size-regression tripwire's sibling).
MAX_BLOCK_LINES = 240

#: S1 discipline 3.
SIZE_REGRESSION_THRESHOLD = 0.40
SIZE_REGRESSION_MIN_LINES = 100

COORDINATED_FAMILIES = ("coordinated_edit_set",)

#: THE FIXER SEAM — LANDED 2026-08-29, kept here as the contract.
#:
#: The family name is deliberately NOT the census shape ``multi_line_coordinated``.
#: That token is a member of ``fixer.COMPLEX_CLASSES``, whose invariant is that no
#: census shape may name a FAMILY: a class token that names a family routes the
#: whole fix stage into that ONE enumerator, and this shape has no single
#: enumerator that expresses it — it must fall through to the union of every
#: enabled family, of which the coordinated family is one member.
#:
#: ``test_edit_sets_offline.py::test_fixer_grants_and_applies_the_coordinated_family``
#: and ``::test_fixer_seam_is_documented_or_applied`` hold all three halves
#: together: a grant with no apply seam would let the family be SELECTED and then
#: silently applied as its first line only.
FIXER_SEAM = """
1. fixer.py FAMILY_MAP (the ONLY grant path — the 1B's A-E class menu can never
   name this family, so it reaches the menu exclusively through the union arm):

       "coordinated_edit_set": "coordinated_edit_set",

2. gym_config.py fix.families default list: append "coordinated_edit_set".

3. fixer.py application — the THREE candidate-application sites (witness_replay
   pick, spec_exec pick, mutation-menu pick) become one call, which is
   byte-identical to `c2 = list(lines); c2[li] = new` for a plain 3-tuple:

       import edit_sets
       c2 = edit_sets.apply_candidate(lines, mut)      # None => skip, fail closed

   `apply_candidate` returns None rather than a partially-applied list, so the
   existing `if c2 is None: continue` / SyntaxError fall-through covers it.
"""


# ============================================================ 1. THE SUBSTRATE
class SpanEdit:
    """One exact-match span replacement. ``new == ""`` deletes the span."""

    __slots__ = ("old", "new", "role")

    def __init__(self, old, new="", role=""):
        self.old = old
        self.new = new
        self.role = role

    def __eq__(self, other):
        return (isinstance(other, SpanEdit) and self.old == other.old
                and self.new == other.new)

    def __hash__(self):
        return hash((self.old, self.new))

    def __repr__(self):                                   # pragma: no cover
        return "SpanEdit(%r -> %r, role=%r)" % (self.old[:40], self.new[:40],
                                                self.role)


class EditSetError(Exception):
    """A span edit could not be anchored. Always fatal for the WHOLE set."""


def base_digest(source):
    """Stable identity of the snapshot a set was generated against."""
    import hashlib
    return hashlib.sha256((source or "").encode("utf-8")).hexdigest()[:16]


def apply_span_edits(source, edits, expect_base=None):
    """Apply every edit SIMULTANEOUSLY against one immutable base, or ``None``.

    S1's four disciplines plus the order-independence the w30 set-level
    arbitration edges answer demands. ``None`` (never a partial string) is the
    only failure signal, so a caller cannot mistake half a set for a candidate.

      1. exact byte match, no normalization, no fuzzy fallback -> 0 matches is
         a hard reject;
      2. uniqueness required -> >1 match is a hard reject, never guessed;
      3. size-regression tripwire;
      4. parse gate, strict (deliberately NOT the executor's fail-open one).

    ORDER-INDEPENDENCE (w30 "order-dependent edits" / "overlapping edits"):
    every anchor is resolved to a byte range on the BASE snapshot, write-write
    overlaps are rejected, and the ranges are spliced in one pass. No edit can
    therefore target text an earlier edit created, and permuting the edit list
    cannot change the result. That is why `p2_hoist_shared_subexpr` may safely
    re-introduce the canonical text inside the hoisted binding: the consumer
    anchors were already resolved against the original.

    COMPARE-AND-SWAP (w30 "mixed-base set" / "duplicate retry"): pass
    `expect_base` and the apply is refused unless the snapshot still digests to
    it, which also makes a second application of the same set a rejection rather
    than a silently different tree.
    """
    if not edits:
        return None
    if expect_base is not None and base_digest(source) != expect_base:
        return None
    ranges = []
    for ed in edits:
        if not isinstance(ed, SpanEdit) or not ed.old:
            return None
        if ed.old == ed.new:
            return None                      # a no-op edit is not an edit
        count = source.count(ed.old)
        if count == 0:
            return None                      # 1. exact match, fail closed
        if count > 1:
            return None                      # 2. uniqueness required
        start = source.index(ed.old)
        ranges.append((start, start + len(ed.old), ed.new))
    ranges.sort()
    for (s1, e1, _), (s2, _e2, _n) in zip(ranges, ranges[1:]):
        if s2 < e1:
            return None                      # write-write overlap: never merged
    out, cursor = [], 0
    for start, end, new in ranges:
        out.append(source[cursor:start])
        out.append(new)
        cursor = end
    out.append(source[cursor:])
    out = "".join(out)
    if _size_regressed(source, out):
        return None                          # 3. size-regression tripwire
    if not _parses(out):                     # 4. parse gate (strict, never open)
        return None
    return out


def _parses(text):
    """Strict parse gate, with ONE concession: a uniformly indented span.

    A span lifted out of a class body or a closure (`invoke_fallback` in the v3
    board is a nested `def`) is not a module and never parses as one. Dedenting
    once is exactly what `_parse_span` already does to READ such a span, so the
    WRITE gate has to accept the same shape or the whole family is silently
    unavailable inside every nested function. It is not a fail-open: text that
    parses neither way is still rejected, and a set that broke the indentation
    structure fails `textwrap.dedent` + `ast.parse` just as loudly.
    """
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        pass
    except Exception:
        return False
    import textwrap
    try:
        ast.parse(textwrap.dedent(text))
        return True
    except Exception:
        return False


def _size_regressed(before, after):
    n = before.count("\n") + 1
    if n < SIZE_REGRESSION_MIN_LINES:
        return False
    return len(after) < len(before) * SIZE_REGRESSION_THRESHOLD


class EditSet(tuple):
    """A coordinated candidate that IS a ``(li, old_block, new_block)`` tuple.

    Extra state rides as attributes, so every existing 3-tuple consumer is
    unaffected and only `apply_candidate` needs to know the difference.
    """

    # NOTE: no __slots__ -- CPython forbids a non-empty __slots__ on a tuple
    # subclass (variable-length instance layout). The attributes live in a
    # per-instance dict; the tuple half is still immutable, which is the part
    # every downstream consumer depends on.

    def __new__(cls, line_span, old_block, new_block, edits, pattern, coupling):
        self = tuple.__new__(cls, (line_span[0], old_block, new_block))
        self.line_span = (line_span[0], line_span[1])
        self.edits = tuple(edits)
        self.pattern = pattern
        self.coupling = tuple(sorted(set(coupling)))
        return self

    @property
    def n_edits(self):
        return len(self.edits)

    @property
    def n_lines(self):
        return self.line_span[1] - self.line_span[0] + 1

    def describe(self):
        return "%s: %d edits over lines %d-%d (coupled on %s)" % (
            self.pattern, self.n_edits, self.line_span[0] + 1,
            self.line_span[1] + 1, ", ".join(self.coupling) or "-")


def apply_candidate(lines, mut):
    """THE SEAM. New line list, or ``None``.

    A plain ``(li, old, new)`` tuple takes the exact old path; an `EditSet`
    splices its contiguous block. Fail-closed on an anchor mismatch either way.
    """
    try:
        li, old, new = mut
    except (TypeError, ValueError):
        return None
    span = getattr(mut, "line_span", None)
    out = list(lines)
    if span is None:
        if not (0 <= li < len(out)):
            return None
        out[li] = new
        return out
    lo, hi = span
    if not (0 <= lo <= hi < len(out)):
        return None
    if "\n".join(out[lo:hi + 1]) != old:
        return None
    out[lo:hi + 1] = new.split("\n")
    return out


def apply_to_span(span_lines, mut):
    """Span-relative application — ANCHORED BY TEXT, never by index.

    Candidate line indices are absolute file offsets while `span_lines` is
    span-relative (the contract `witness_replay.family_witness` and
    `intent_oracle._apply` already work under), so the anchor is the OLD BLOCK's
    text. Ambiguity is a rejection, not a guess: a wrong anchor could manufacture
    a false kill, and that is exactly the drift S1's uniqueness rule forbids.
    """
    try:
        _li, old, new = mut
    except (TypeError, ValueError):
        return None
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    if getattr(mut, "line_span", None) is None:
        hits = [i for i, ln in enumerate(span_lines) if ln == old]
        if len(hits) != 1:
            return None
        out = list(span_lines)
        out[hits[0]] = new
        return out
    block = old.split("\n")
    n = len(block)
    hits = [i for i in range(len(span_lines) - n + 1)
            if span_lines[i:i + n] == block]
    if len(hits) != 1:
        return None
    out = list(span_lines)
    out[hits[0]:hits[0] + n] = new.split("\n")
    return out


# ============================================================ 2. COUPLING
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _symbols(text):
    return set(_NAME_RE.findall(text or ""))


def coupling_ok(edits):
    """Every edit must share at least one symbol with the rest of the set.

    A cheap independent re-assertion of what the generators build by
    construction: an orphan atom never enters a set just because it scored well
    on its own.
    """
    if len(edits) < 2:
        return True
    texts = [(_symbols(e.old) | _symbols(e.new)) for e in edits]
    for i, syms in enumerate(texts):
        others = set()
        for j, o in enumerate(texts):
            if i != j:
                others |= o
        if not (syms & others):
            return False
    return True


def _mk_set(span_lines, start_idx, edits, pattern, coupling):
    """Realize span edits into a contiguous-block `EditSet`, or ``None``.

    Every gate that can reject the set runs HERE, so no generator can emit a
    set that would not apply: coupling, envelope, atomic application against the
    unmodified span, and the parse gate inside `apply_span_edits`.
    """
    if not edits or len(edits) > MAX_SET_EDITS:
        return None
    if not coupling_ok(edits):
        return None
    src = "\n".join(span_lines)
    applied = apply_span_edits(src, edits)
    if applied is None:
        return None
    new_lines = applied.split("\n")
    lo, hi = _changed_envelope(span_lines, new_lines)
    if lo is None:
        return None                                  # no-op set: never emitted
    if (hi - lo + 1) > MAX_BLOCK_LINES:
        return None
    changed, hunks = diff_shape(span_lines, new_lines)
    if changed > MAX_CHANGED_LINES or hunks > MAX_HUNKS:
        return None
    # the replacement block is everything the new file holds between the same
    # boundaries, measured from the END so a length change lands inside it
    tail = len(span_lines) - 1 - hi
    new_hi = len(new_lines) - 1 - tail
    if new_hi < lo:
        new_hi = lo - 1                              # pure deletion
    old_block = "\n".join(span_lines[lo:hi + 1])
    new_block = "\n".join(new_lines[lo:new_hi + 1])
    if old_block == new_block:
        return None
    return EditSet((start_idx + lo, start_idx + hi), old_block, new_block,
                   edits, pattern, coupling)


def diff_shape(old_lines, new_lines):
    """(changed_lines, hunks) — the census's own diff-shape prior, measured."""
    import difflib
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    changed = hunks = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1
        changed += max(i2 - i1, j2 - j1)
    return changed, hunks


def _changed_envelope(old_lines, new_lines):
    """(first, last) index in OLD lines that the rewrite touches."""
    n_old, n_new = len(old_lines), len(new_lines)
    lo = 0
    while lo < n_old and lo < n_new and old_lines[lo] == new_lines[lo]:
        lo += 1
    if lo == n_old and n_old == n_new:
        return None, None
    i, j = n_old - 1, n_new - 1
    while i >= lo and j >= lo and old_lines[i] == new_lines[j]:
        i -= 1
        j -= 1
    return lo, max(i, lo)


# ============================================================ 3. AST HELPERS
def _parse_span(span_lines):
    src = "\n".join(span_lines)
    try:
        return ast.parse(src), src
    except SyntaxError:
        # a function span lifted out of a class body is indented; dedent once
        import textwrap
        try:
            ded = textwrap.dedent(src)
            return ast.parse(ded), ded
        except SyntaxError:
            return None, src


def _seg(span_lines, node):
    """The exact source text of a node, as whole lines (so it is a unique span)."""
    lo = node.lineno - 1
    hi = getattr(node, "end_lineno", node.lineno) - 1
    if lo < 0 or hi >= len(span_lines):
        return None
    return "\n".join(span_lines[lo:hi + 1])


def _expr_text(span_lines, node):
    """The exact source text of an expression node (may be mid-line)."""
    lo = node.lineno - 1
    hi = getattr(node, "end_lineno", node.lineno) - 1
    if lo < 0 or hi >= len(span_lines):
        return None
    if lo == hi:
        return span_lines[lo][node.col_offset:node.end_col_offset]
    parts = [span_lines[lo][node.col_offset:]]
    parts += span_lines[lo + 1:hi]
    parts.append(span_lines[hi][:node.end_col_offset])
    return "\n".join(parts)


def _indent_of(line):
    return line[:len(line) - len(line.lstrip())]


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and \
        isinstance(node.value, str) else None


# ============================================================ 4. THE PATTERNS
#
# Five coordination patterns were committed in /tmp/solq/w27/a_ml_pertask.md.
# v1 implements the three whose gold edit-set is DERIVABLE from tokens the
# parent source already contains. The two remaining patterns are declared here
# rather than silently missing, with the measured reason each is punted — see
# `PUNTED_PATTERNS` and the reachability report.

def p1_authority_thread(span_lines, start_idx, file_lines=None):
    """P1 — canonical authority/scope threaded into guard + write.

    Seed: a keyword or assignment writing ``RECV.ATTR`` where another in-scope
    object also carries ``ATTR``. The coupled sibling is the existing
    ``if not RECV:`` guard, strengthened to cross-check the two objects' ATTR.
    Both atoms name RECV and ATTR, which is the coupling.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    # The alternate authorities are the span's ENTITY names: bound from a
    # row-returning query AND existence-guarded. Deliberately not "any name
    # that happens to carry the attribute" -- the authoritative object usually
    # has not been read for that attribute yet, which is the whole bug.
    owners = _entity_names(tree, span_lines)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg is None:
            continue
        val = node.value
        if not (isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name)):
            continue
        recv, attr = val.value.id, val.attr
        if node.arg != attr:                       # `workspace_id=X.workspace_id`
            continue
        write_txt = _expr_text(span_lines, val)
        if not write_txt:
            continue
        if recv not in owners:
            continue
        for alt in sorted(owners):
            if alt == recv:
                continue
            # ATOM B: retarget the write to the authoritative object
            line = span_lines[val.lineno - 1]
            if line.count(write_txt) != 1:
                continue
            edit_b = SpanEdit(line, line.replace(write_txt, "%s.%s" % (alt, attr)),
                              role="retarget_write")
            # ATOM A: strengthen the old receiver's existence guard so the two
            # objects' ATTR must agree before the write is authorized
            guard = _find_truthiness_guard(span_lines, recv)
            if guard is None:
                continue
            gi, gline = guard
            if gline == line:
                continue
            ind = _indent_of(gline)
            new_guard = ("%sif not %s or %s.%s is None or %s.%s != %s.%s:"
                         % (ind, recv, recv, attr, recv, attr, alt, attr))
            edit_a = SpanEdit(gline, new_guard, role="strengthen_guard")
            es = _mk_set(span_lines, start_idx, [edit_a, edit_b],
                         "authority_thread", [recv, alt, attr])
            if es is not None:
                out.append(es)
    return out


_ROW_TERMINALS = frozenset(("first", "one", "one_or_none", "scalar", "get"))


def _entity_names(tree, span_lines):
    """Names bound from a row-returning query AND existence-guarded."""
    bound = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                and call.func.attr in _ROW_TERMINALS:
            bound.add(node.targets[0].id)
    guarded = set()
    for ln in span_lines:
        m = re.match(r"^\s*if\s+not\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(#.*)?$", ln)
        if m:
            guarded.add(m.group(1))
    return bound & guarded


def _find_truthiness_guard(span_lines, name):
    pat = re.compile(r"^\s*if\s+not\s+%s\s*:\s*(#.*)?$" % re.escape(name))
    hits = [(i, ln) for i, ln in enumerate(span_lines) if pat.match(ln)]
    return hits[0] if len(hits) == 1 else None


def p2_hoist_shared_subexpr(span_lines, start_idx, file_lines=None):
    """P2 — normalize/filter once, reuse at every consumer.

    Seed: two or more expressions in the span that are built from the SAME
    base sub-expression ``OBJ.get("KEY")``. Hoist the STRICTEST consumer (the
    one with the most AST nodes — the one that already filters) to a binding
    named after the key, then rewrite every consumer to that name. The coupling
    is the key itself: no consumer may bypass the filtered source.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    src = "\n".join(span_lines)
    # group the seed calls by the (obj, key) they derive from
    groups = {}
    for node in ast.walk(fn):
        base = _get_call_base(node)
        if base is None:
            continue
        groups.setdefault(base, []).append(node)
    out = []
    for (obj, key), seeds in sorted(groups.items()):
        if not key.isidentifier() or len(seeds) < 2:
            continue
        if key in _bound_names(fn):
            continue                              # the name is already taken
        parents = _parent_map(fn)
        seeds = sorted(seeds, key=lambda n: (n.lineno, n.col_offset))
        # Two climbs per seed. INNER stops at a comprehension boundary (the
        # expression that *feeds* an iteration); OUTER climbs through it (the
        # whole normalization, filter and slice included). One candidate set per
        # choice of canonical seed, first occurrence first -- the choice is a
        # menu entry, not a guess the enumerator has to get right.
        inner, outer = [], []
        ok = True
        for s in seeds:
            i_txt = _expr_text(span_lines, _climb(parents, s, through_comp=False))
            o_txt = _expr_text(span_lines, _climb(parents, s, through_comp=True))
            if not i_txt or not o_txt:
                ok = False
                break
            inner.append(i_txt)
            outer.append(o_txt)
        if not ok:
            continue
        for ci in range(len(seeds)):
            canonical = outer[ci]
            texts = [canonical if j == ci else inner[j] for j in range(len(seeds))]
            if len(set(texts)) != len(texts):
                continue
            if any(src.count(t) != 1 for t in texts):
                continue
            anchor_i = _hoist_anchor(fn, seeds[0].lineno)
            if anchor_i is None:
                continue
            aline = span_lines[anchor_i]
            if src.count(aline + "\n") != 1:
                continue
            ind = _indent_of(aline)
            # CONSUMERS FIRST, HOIST LAST: the hoisted line re-introduces the
            # canonical text, so hoisting first would make the consumer edit's
            # anchor ambiguous and (correctly) fail the uniqueness discipline.
            edits = [SpanEdit(t, key, role="retarget_consumer")
                     for t in sorted(texts, key=lambda t: (-len(t), t))]
            edits.append(SpanEdit(
                aline + "\n",
                "%s\n%s%s = %s\n" % (aline, ind, key, canonical), role="hoist"))
            es = _mk_set(span_lines, start_idx, edits, "hoist_shared_subexpr",
                         [obj, key])
            if es is not None:
                out.append(es)
    return out


def _parent_map(fn):
    parents = {}
    for p in ast.walk(fn):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    return parents


def _climb(parents, node, through_comp):
    """Maximal enclosing expression, stopping at a container-literal boundary."""
    cur = node
    while True:
        par = parents.get(id(cur))
        if par is None:
            return cur
        if isinstance(par, ast.comprehension):
            if not through_comp:
                return cur
            # the comprehension itself is not an expression; step to its owner
            owner = parents.get(id(par))
            if owner is None:
                return cur
            cur = owner
            continue
        if not isinstance(par, ast.expr):
            return cur
        if isinstance(par, (ast.Dict, ast.Tuple, ast.List, ast.Set)):
            return cur
        if isinstance(par, ast.Call) and par.func is not cur:
            return cur
        cur = par


def _get_call_base(node):
    """``OBJ.get("KEY")`` -> ``(OBJ, KEY)``."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if not (isinstance(f, ast.Attribute) and f.attr == "get"
            and isinstance(f.value, ast.Name) and len(node.args) == 1):
        return None
    key = _const_str(node.args[0])
    return (f.value.id, key) if key else None


def _bound_names(fn):
    names = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
    return names


def _hoist_anchor(fn, before_line):
    """Index of the LAST line of the last top-level simple statement above the use.

    Anchoring on a whole statement (never a bare line) keeps the inserted
    binding at a legal statement boundary and at the body's own indentation.
    """
    best = None
    for st in fn.body:
        if st.lineno >= before_line:
            break
        if isinstance(st, (ast.Assign, ast.AnnAssign, ast.Expr)):
            best = st
    if best is None:
        return None
    return getattr(best, "end_lineno", best.lineno) - 1


def p3_dict_snapshot_to_accessor(span_lines, start_idx, file_lines=None):
    """P3 — access-path migration: kill a ``__dict__`` snapshot, read descriptors.

    Seed: ``NAME = getattr(OBJ, "__dict__", ...)``. Every ``NAME.get(K)`` in the
    function becomes ``getattr(OBJ, K, None)`` and the snapshot binding is
    DELETED (a span edit with ``new == ""`` — free, and the line-tuple shape
    could not express it). Coupling: NAME and OBJ.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    out = []
    for st in fn.body:
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)):
            continue
        call = st.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "getattr" and len(call.args) >= 2):
            continue
        if _const_str(call.args[1]) != "__dict__":
            continue
        if not isinstance(call.args[0], ast.Name):
            continue
        name, obj = st.targets[0].id, call.args[0].id
        uses = [n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == name and len(n.args) == 1]
        if not uses:
            continue
        src = "\n".join(span_lines)
        edits, ok = [], True
        for u in uses:
            t = _expr_text(span_lines, u)
            karg = _expr_text(span_lines, u.args[0])
            if t is None or karg is None or src.count(t) != 1:
                ok = False
                break
            edits.append(SpanEdit(t, "getattr(%s, %s, None)" % (obj, karg),
                                  role="retarget_read"))
        if not ok:
            continue
        bind = _seg(span_lines, st)
        if bind is None or src.count(bind + "\n") != 1:
            continue
        edits.append(SpanEdit(bind + "\n", "", role="drop_snapshot"))
        es = _mk_set(span_lines, start_idx, edits, "dict_snapshot_to_accessor",
                     [name, obj])
        if es is not None:
            out.append(es)
    return out


def _first_function(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


# ==================================================== 4b. ASSEMBLY: WRAPPERS
#
# The ASSEMBLY class from `PUNTED_PATTERNS`: every token of the gold set already
# exists in the parent FILE, and what was missing was a generator that composes
# them. ML-06's gold is the canonical member — it replaces a call to the local
# `invoke(...)` wrapper with that wrapper's own body, the call site's actual
# arguments substituted in, at ONE branch of a preceding `if`.
#
# Nothing below is specialized to that task. Two mechanisms, each independently
# tested on synthetic fixtures in `test_edit_sets_offline.py`:
#
#   MINE     a sibling function whose whole body is ``return <call>`` is a
#            wrapper; its parameters are the substitution slots.
#   INLINE   substituting the call site's argument EXPRESSIONS (as AST, never as
#            text) into that return expression is the inlining.
#
# and one composition, `p5_branch_specialized_call`, which is where the
# coordination lives: a call that follows an `if` whose body rebinds names the
# call reads can be specialized INTO both branches, because inside the branch
# those names have known values and outside it they do not.
#
# Free variables are the correctness risk of any inliner. A mined wrapper is
# only offered at a call site it could already legally reach: the wrapper must be
# defined in a scope that ENCLOSES the span (checked structurally against the
# file AST), so every non-parameter name in its body resolves at the call site to
# exactly what it resolved to inside the wrapper.

#: Per-generator emission caps. These are budgets on the MENU, not truncations of
#: a search: each generator walks a bounded seed set (wrapper call sites / if-then
#: -call adjacencies) and both are small by construction. Reaching one is still
#: reported through the shared `truncated` metadata.
MAX_INLINE_CANDIDATES = 2
MAX_BRANCH_CANDIDATES = 6


class WrapperSpec:
    """A sibling function that is nothing but ``return <call>(...)``."""

    __slots__ = ("name", "params", "ret", "lineno", "end_lineno")

    def __init__(self, name, params, ret, lineno, end_lineno):
        self.name = name
        self.params = tuple(params)
        self.ret = ret
        self.lineno = lineno
        self.end_lineno = end_lineno

    def __repr__(self):                                   # pragma: no cover
        return "WrapperSpec(%s/%d)" % (self.name, len(self.params))


def _wrapper_of(node):
    """`WrapperSpec` if `node` is a single-``return``-call function, else None."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and _const_str(body[0].value) is not None:
        body = body[1:]                                   # skip the docstring
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    ret = body[0].value
    if not isinstance(ret, ast.Call):
        return None
    a = node.args
    if a.vararg or a.kwarg or a.kwonlyargs:
        return None                    # */** slots are not positional-substitutable
    params = [p.arg for p in list(getattr(a, "posonlyargs", [])) + list(a.args)]
    if not params or len(set(params)) != len(params):
        return None
    return WrapperSpec(node.name, params, ret, node.lineno,
                       getattr(node, "end_lineno", node.lineno))


def mine_wrappers(file_lines, span_lo=None, span_hi=None):
    """Wrappers defined in a scope that ENCLOSES ``[span_lo, span_hi]``.

    The enclosure test is what makes inlining safe: a wrapper's free variables
    are only guaranteed to resolve identically at the call site when the call
    site sits inside the same scope the wrapper does. A wrapper from an unrelated
    function is never offered, which is a hard skip, not a ranking penalty.
    """
    if not file_lines:
        return {}
    try:
        tree = ast.parse("\n".join(file_lines))
    except SyntaxError:
        return {}
    out = {}
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        if span_lo is not None and isinstance(parent, (ast.FunctionDef,
                                                       ast.AsyncFunctionDef)):
            plo = parent.lineno
            phi = getattr(parent, "end_lineno", parent.lineno)
            if not (plo <= span_lo + 1 and span_hi + 1 <= phi):
                continue               # this scope does not enclose the span
        elif span_lo is not None and not isinstance(parent, ast.Module):
            continue
        for st in body:
            spec = _wrapper_of(st)
            if spec is not None and spec.name not in out:
                out[spec.name] = spec
    return out


class _Substituter(ast.NodeTransformer):
    """Replace bare ``Name`` loads by a mapping of name -> replacement AST."""

    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            import copy
            return copy.deepcopy(self.mapping[node.id])
        return node


def inline_wrapper_call(spec, call):
    """The wrapper's body with this call's arguments substituted, or ``None``.

    Substitution is over the AST, so a textual collision (a parameter name that
    also occurs inside a string, or an argument that would need parenthesising)
    cannot corrupt the result. Anything the slot model does not express exactly —
    a star-arg at the call site, an unknown keyword, a missing argument — is a
    rejection rather than a partial inline.
    """
    import copy
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    if any(k.arg is None for k in call.keywords):
        return None
    if len(call.args) > len(spec.params):
        return None
    mapping = {}
    for i, arg in enumerate(call.args):
        mapping[spec.params[i]] = arg
    for kw in call.keywords:
        if kw.arg not in spec.params or kw.arg in mapping:
            return None
        mapping[kw.arg] = kw.value
    if set(mapping) != set(spec.params):
        return None                     # a defaulted slot is not modelled: skip
    return _Substituter(mapping).visit(copy.deepcopy(spec.ret))


def _unparse(node):
    try:
        return ast.unparse(node)
    except Exception:                                     # pragma: no cover
        return None


def _call_of(stmt):
    """``(target_text, call_node)`` for ``X = f(...)`` / ``f(...)``, else None."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
            and isinstance(stmt.value, ast.Call):
        t = _unparse(stmt.targets[0])
        return (t, stmt.value) if t else None
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return (None, stmt.value)
    return None


def _render_stmt(indent, target, expr):
    txt = _unparse(expr)
    if txt is None:
        return None
    return "%s%s%s" % (indent, ("%s = " % target) if target else "", txt)


def _stmt_lists(fn):
    """Every statement list inside a function, outermost first."""
    out = []
    stack = [fn]
    while stack:
        node = stack.pop(0)
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(node, field, None)
            if isinstance(seq, list) and seq and all(isinstance(s, ast.stmt)
                                                     for s in seq):
                if not (node is fn and field != "body"):
                    out.append(seq)
                for s in seq:
                    if isinstance(s, (ast.If, ast.Try, ast.For, ast.While,
                                      ast.With, ast.AsyncWith, ast.AsyncFor)):
                        stack.append(s)
                    # EXCEPT HANDLERS ARE REACHABLE CODE. Missing them would not
                    # merely lose a seed — `p5` asks this walk which names are
                    # read AFTER a call, and a name read only in a handler would
                    # then look dead and be folded away. That is a behaviour
                    # change wearing a refactor's clothes.
                    if isinstance(s, ast.Try):
                        stack.extend(s.handlers)
    return out


def p4_wrapper_inline(span_lines, start_idx, file_lines=None):
    """P4 — replace a call to a mined wrapper with the wrapper's own body.

    Pattern-general assembly: no token is invented, every one comes from the
    wrapper's return expression and the call site's own arguments. Coupling is
    the wrapper name plus the callee it forwards to — the two halves of the
    indirection this edit removes.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    span_lo, span_hi = start_idx, start_idx + len(span_lines) - 1
    wrappers = mine_wrappers(file_lines, span_lo, span_hi) if file_lines else {}
    if not wrappers:
        wrappers = {}
        for node in ast.walk(tree):
            spec = _wrapper_of(node)
            if spec is not None and spec.name != fn.name:
                wrappers.setdefault(spec.name, spec)
    out = []
    for sl in _stmt_lists(fn):
        for st in sl:
            got = _call_of(st)
            if got is None:
                continue
            target, call = got
            if not isinstance(call.func, ast.Name):
                continue
            spec = wrappers.get(call.func.id)
            if spec is None or spec.name == fn.name:
                continue
            inlined = inline_wrapper_call(spec, call)
            if inlined is None:
                continue
            lo = st.lineno - 1
            hi = getattr(st, "end_lineno", st.lineno) - 1
            if lo < 0 or hi >= len(span_lines):
                continue
            old = "\n".join(span_lines[lo:hi + 1])
            new = _render_stmt(_indent_of(span_lines[lo]), target, inlined)
            if new is None or new == old:
                continue
            callee = getattr(spec.ret.func, "id",
                             getattr(spec.ret.func, "attr", ""))
            es = _mk_set(span_lines, start_idx,
                         [SpanEdit(old, new, role="inline_wrapper")],
                         "wrapper_inline", [spec.name, callee])
            if es is not None:
                out.append(es)
            if len(out) >= MAX_INLINE_CANDIDATES:
                return out
    return out


def _none_initialized(fn):
    """Names bound to a literal ``None`` somewhere in the function."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) \
                and n.value.value is None:
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and isinstance(n.value, ast.Constant) and n.value.value is None:
            out.add(n.target.id)
    return out


def _has_effect(node):
    """True if evaluating this expression could do something observable."""
    return any(isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom,
                              ast.NamedExpr))
               for n in ast.walk(node))


class _BranchSpecializer(ast.NodeTransformer):
    """Rewrite an expression under what a branch is KNOWN to have established.

    Two rules, both plain dataflow rather than heuristics:

      * a name the branch rebinds to `EXPR` and that nothing reads after the call
        is, inside the branch, exactly `EXPR` (and outside it, untouched);
      * ``A or B`` where `A` starts as ``None`` and the branch assigns it is `A`
        inside the branch and `B` outside it.
    """

    def __init__(self, inline_map, truthy, still_none):
        self.inline_map = inline_map
        self.truthy = truthy
        self.still_none = still_none

    def visit_BoolOp(self, node):
        # The `or` is resolved against the ORIGINAL head name, before any
        # substitution rewrites it — otherwise folding the head first would hide
        # the very fact that decides which arm of the `or` survives.
        if isinstance(node.op, ast.Or) and len(node.values) == 2 \
                and isinstance(node.values[0], ast.Name):
            head = node.values[0].id
            if head in self.truthy:
                return self.visit(node.values[0])
            if head in self.still_none:
                return self.visit(node.values[1])
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.inline_map:
            import copy
            return copy.deepcopy(self.inline_map[node.id])
        return node


def p5_branch_specialized_call(span_lines, start_idx, file_lines=None):
    """P5 — push a trailing call INTO both arms of the `if` that precedes it.

    Seed: an ``if`` with no ``else`` immediately followed by a call statement
    that reads names the branch rebinds. Inside the branch those names have known
    values; outside it they keep the ones they were bound to above. Specializing
    the call into each arm therefore says something the single shared call could
    not, which is why the branch-only work (a freshly resolved client, a
    just-opened handle) becomes visible instead of being laundered through a
    variable.

    Coordination, not one edit: the branch's now-dead binding is DELETED, the
    call is duplicated, and each copy's arguments are resolved differently. Each
    variant additionally offered with the wrapper inlined (P4's mechanism) in the
    specialized arm, and in both arms — the choice of how far to inline is a menu
    entry, never a guess the enumerator has to get right.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    span_lo, span_hi = start_idx, start_idx + len(span_lines) - 1
    wrappers = mine_wrappers(file_lines, span_lo, span_hi) if file_lines else {}
    none_init = _none_initialized(fn)
    out = []
    for sl in _stmt_lists(fn):
        for i in range(len(sl) - 1):
            st, nxt = sl[i], sl[i + 1]
            if not isinstance(st, ast.If) or st.orelse:
                continue
            got = _call_of(nxt)
            if got is None:
                continue
            target, call = got
            # what the branch establishes
            assigned = {}
            for b in st.body:
                if isinstance(b, ast.Assign) and len(b.targets) == 1 \
                        and isinstance(b.targets[0], ast.Name):
                    assigned[b.targets[0].id] = b
            if not assigned:
                continue
            after = set()
            for later in sl[i + 2:]:
                after |= {n.id for n in ast.walk(later) if isinstance(n, ast.Name)}
            for other in _stmt_lists(fn):
                if other is sl:
                    continue
                for o in other:
                    if o.lineno <= nxt.lineno:
                        continue
                    after |= {n.id for n in ast.walk(o)
                              if isinstance(n, ast.Name)}
            # Only a binding nothing reads after the call may be folded away,
            # and only one whose value is SIDE-EFFECT FREE: substituting a name
            # into an expression can duplicate it (`x or f(x)`) or move it past
            # another operand, and neither is allowed to change how many times a
            # call runs. A binding whose RHS calls something therefore stays put
            # and keeps its line — the specialization above it is still real.
            inline_map = {k: v.value for k, v in assigned.items()
                          if k not in after and not _has_effect(v.value)}
            truthy = set(assigned) & none_init
            still_none = truthy - set(inline_map)
            if not inline_map and not truthy:
                continue
            import copy
            if_call = _BranchSpecializer(inline_map, truthy,
                                         set()).visit(copy.deepcopy(call))
            else_call = _BranchSpecializer({}, set(),
                                           still_none).visit(copy.deepcopy(call))
            if _unparse(if_call) == _unparse(else_call):
                continue                       # no specialization: not this seed
            lo = st.lineno - 1
            hi = getattr(nxt, "end_lineno", nxt.lineno) - 1
            if lo < 0 or hi >= len(span_lines):
                continue
            head_ind = _indent_of(span_lines[lo])
            body_lo = st.body[0].lineno - 1
            body_ind = _indent_of(span_lines[body_lo])
            dropped = {a.lineno - 1 for k, a in assigned.items()
                       if k in inline_map}
            kept = [span_lines[j] for j in range(lo, body_lo)]
            for j in range(body_lo, nxt.lineno - 1):
                if any(a.lineno - 1 <= j <= getattr(a, "end_lineno", a.lineno) - 1
                       for k, a in assigned.items() if k in inline_map):
                    continue
                kept.append(span_lines[j])
            old = "\n".join(span_lines[lo:hi + 1])
            spec = None
            if isinstance(call.func, ast.Name):
                spec = wrappers.get(call.func.id)
            # (a) specialize only, (b) + inline the specialized arm, (c) + both
            variants = [(if_call, else_call, "branch_specialized_call")]
            if spec is not None:
                a_in = inline_wrapper_call(spec, if_call)
                b_in = inline_wrapper_call(spec, else_call)
                if a_in is not None:
                    variants.append((a_in, else_call,
                                     "branch_specialized_inline"))
                    if b_in is not None:
                        variants.append((a_in, b_in, "branch_specialized_inline"))
            for a_expr, b_expr, pattern in variants:
                a_line = _render_stmt(body_ind, target, a_expr)
                b_line = _render_stmt(body_ind, target, b_expr)
                if a_line is None or b_line is None:
                    continue
                new = "\n".join(kept + [a_line, head_ind + "else:", b_line])
                if new == old:
                    continue
                coupling = [c for c in (target, getattr(call.func, "id", None),
                                        spec.name if spec else None) if c]
                coupling += sorted(inline_map) or sorted(truthy)
                es = _mk_set(span_lines, start_idx,
                             [SpanEdit(old, new, role="specialize_branches")],
                             pattern, coupling)
                if es is not None:
                    out.append(es)
                if len(out) >= MAX_BRANCH_CANDIDATES:
                    return out
    return out


#: PUNTS, DECLARED RATHER THAN SILENTLY MISSING (a family with no grant path is
#: a dead family, #181 — so is a pattern nobody wrote down).
#:
#: Every rationale below is MEASURED against the parent file, not reasoned:
#: `probes/ml_reachability.py` prints them beside the table, and
#: `test_edit_sets_offline.py::test_punt_rationales_are_measured` re-checks the
#: token claims against the corpus so a stale excuse cannot survive a fix.
#:
#: Two distinct causes, and they need different work:
#:   NOVEL-TOKEN   a token in the gold set occurs NOWHERE in the parent file, so
#:                 no enumerator over the parent can reach it at all. This is a
#:                 hard expressiveness boundary, not a pruning failure.
#:   ASSEMBLY      every token exists in the parent file, but composing them into
#:                 the gold shape needs a generator nobody has written yet. These
#:                 are reachable in principle and are the next build order.
PUNTED_PATTERNS = {
    # branch_specialized_fallback: NO LONGER PUNTED (2026-08-29). It was the one
    # ASSEMBLY-class entry — every token of ML-06's gold already existed in the
    # parent file — and `p5_branch_specialized_call` + `p4_wrapper_inline` now
    # compose it: gold reaches rank 2 of a 4-wide menu. The four remaining punts
    # are all NOVEL-TOKEN, i.e. a hard expressiveness boundary for any enumerator
    # over the parent alone, not a pruning failure that more search would fix.
    "failure_recovery_rewrite": (
        "NOVEL-TOKEN. ML-09/ML-10: gold's escalation `reason`/`question` and "
        "retry `reason` are newly AUTHORED prose strings. The callees "
        "(`_escalate_needs_human`, `_do_retry`) are in the parent file; the "
        "string arguments are not, and AST equality counts them."),
    "authority_derivation_block": (
        "NOVEL-TOKEN. ML-07/ML-08: gold binds invented locals "
        "(`_claimed_author`, `_author_type`, `_created_by`) in a 4-line sanitize "
        "block. `derive_authenticated_actor` IS in the parent file, so the "
        "hoist-and-thread half is reachable; the invented binding names are not."),
    "population_filter_definition": (
        "NOVEL-TOKEN. ML-04: gold invents the name `launched` and the literal "
        "'Turn dispatch failed', neither of which occurs in the parent file. The "
        "retarget half (`recent_tasks` -> `launched` at both sites) is exactly "
        "P2's shape; the DEFINITION half is out of reach."),
    "nested_key_fallback": (
        "NOVEL-TOKEN. ML-03: gold binds an invented local `_cc`. The nesting key "
        "'autopilot' IS minable from the parent file, so only the binding name "
        "blocks it — a generator that reuses an existing name instead of "
        "inventing one would land a behaviourally identical, AST-unequal set."),
}

GENERATORS = (
    ("dict_snapshot_to_accessor", p3_dict_snapshot_to_accessor),
    ("hoist_shared_subexpr", p2_hoist_shared_subexpr),
    ("authority_thread", p1_authority_thread),
    # ASSEMBLY tranche, appended LAST so every menu measured before it keeps its
    # exact order and every gold that was already rank 1 still is.
    ("branch_specialized_call", p5_branch_specialized_call),
    ("wrapper_inline", p4_wrapper_inline),
)


# ================================================ 4c. THE CENSUS FAMILIES
#
# The three shapes the v3 real board carries that no family expressed:
# ``import_change`` (3 tasks), ``statement_insert`` (2) and
# ``exception_handler_change`` (1). They are multi-line INSERTIONS, not line
# rewrites, so they belong on this substrate for the same structural reason the
# coordination patterns do: a `SpanEdit` whose ``new`` is ``stmt + "\n" + old``
# is an insertion, and `_mk_set` already renders that into a contiguous
# ``(line_span, old_block, new_block)`` every downstream consumer understands.
#
# THE EVIDENCE CHANNEL, AND ITS HONEST LIMIT
# ------------------------------------------
# Three designs (a_import_stmt_family.md, statement-insert-enumeration.md,
# a_exc_handler_family.md) all rest on a SIBLING/PROVIDER INDEX: you cannot
# propose `from x import y` unless something told you where `y` lives. There is no
# repo access and assumed that index exists. It did not — `enumerate_mutations`
# was handed the parent FILE and nothing else, and an enumerator over one file
# can only ever re-propose imports that file already has (i.e. bindings that are
# already bound, i.e. never the bug).
#
# So the index is a real, optional, caller-supplied channel: `ProviderIndex`,
# built by `build_provider_index(files)` from the SAME ``files`` mapping
# ``fixer.load_repo`` already returns at the `build_mutation_menu` call site.
# That is the producer (#223: a reader whose writer does not exist is a feature
# that will never run). ``providers=None`` degrades to a file-local index, and
# every generator below is byte-identically empty for callers that pass neither
# — which is why no existing menu moves.

#: Emitted per function by the census tranche, out of its OWN budget so it can
#: neither evict nor reorder a coordination candidate (the same discipline every
#: earlier tranche landed under).
MAX_CENSUS_SETS = 8
#: Explored per function before the census search stops. Hard, and reported.
MAX_CENSUS_EXPLORED = 48
#: Providers retained per undefined binding. Past this it is not evidence.
MAX_PROVIDERS_PER_NAME = 3
#: Undefined bindings a single function may shop for.
MAX_UNDEFINED_NAMES = 4
#: `MAX_TYPES_PER_SITE` / `MAX_WRAP_SITES` / `MAX_WRAP_WIDTH`.
MAX_EXC_TYPES_PER_SITE = 3
MAX_WRAP_SITES = 6
MAX_WRAP_WIDTH = 3
#: "first three boundaries per signal class".
MAX_BOUNDARIES_PER_CLASS = 3
#: Files a provider index will read. A bigger corpus is not a better index, it is
#: a slower one, and the cap is REPORTED (`ProviderIndex.truncated`) rather than
#: silently applied.
MAX_PROVIDER_FILES = 600

_BUILTIN_NAMES = frozenset(dir(__import__("builtins")))


class ProviderIndex:
    """Corpus-mined evidence for the census families. Deterministic.

    Three indexes, all keyed by something a static reader of the BUGGY file can
    ask for:

      ``imports``     binding -> ordered import statements that bind it
      ``exceptions``  exception class name -> how many corpus files name it
      ``handlers``    normalized handler-body template -> support count

    Ordering NEVER depends on dict/hash iteration: every list is sorted by an
    explicit key and every scan walks ``sorted(files)``.
    """

    __slots__ = ("imports", "exceptions", "handlers", "files_scanned",
                 "truncated")

    def __init__(self):
        self.imports = {}
        self.exceptions = {}
        self.handlers = {}
        self.files_scanned = 0
        self.truncated = False

    def providers_for(self, binding):
        """Import statements that bind `binding`, best-supported first."""
        return [stmt for _f, stmt in self.imports.get(binding, ())]

    def is_empty(self):
        return not (self.imports or self.exceptions or self.handlers)


def _import_binding(node, alias):
    if isinstance(node, ast.Import):
        return (alias.asname or alias.name.split(".")[0])
    return (alias.asname or alias.name)


def _import_stmt_text(node, alias):
    """The one-alias rendering of an import statement.

    Deliberately re-rendered from the AST rather than sliced from source: a
    donor `from x import a, b` must not arrive as a provider for `a` carrying
    `b` with it, and a multi-line parenthesised import must not arrive as a
    fragment.
    """
    tail = alias.name + (" as %s" % alias.asname if alias.asname else "")
    if isinstance(node, ast.Import):
        return "import %s" % tail
    dots = "." * (node.level or 0)
    return "from %s%s import %s" % (dots, node.module or "", tail)


def _exc_names(node):
    """Exception class NAMES named by an `except` clause or a `raise`."""
    out = []
    if isinstance(node, ast.ExceptHandler) and node.type is not None:
        targets = (node.type.elts if isinstance(node.type, ast.Tuple)
                   else [node.type])
    elif isinstance(node, ast.Raise) and node.exc is not None:
        exc = node.exc
        targets = [exc.func if isinstance(exc, ast.Call) else exc]
    else:
        return out
    for t in targets:
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, ast.Attribute):
            out.append(t.attr)
    return out


def _handler_shape(handler):
    """A handler body normalized to a template token, or None.

    Only the SMALL vocabulary the design names is recognised, and only shapes
    whose free names are re-derivable at a target site. An arbitrary sibling body
    is deliberately NOT copyable.
    """
    body = [s for s in handler.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if not body:
        return None
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Raise) and s.exc is None:
            return "reraise"
        if isinstance(s, ast.Pass):
            return "pass"
        if isinstance(s, ast.Continue):
            return "continue"
        if isinstance(s, ast.Return):
            if s.value is None:
                return "return_none"
            if isinstance(s.value, ast.Constant):
                return "return_literal"
        return None
    if len(body) == 2 and isinstance(body[1], ast.Raise) and body[1].exc is None \
            and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
        return "log_reraise"
    return None


def build_provider_index(files, exclude=None, max_files=None):
    """A `ProviderIndex` over ``{rel_path: source}``. Deterministic.

    ``exclude`` drops one path (the file under repair — its OWN imports are
    already bound, so it can never be a provider for a name that is undefined in
    it, and including it would let the index "discover" the bug's own absence).
    """
    max_files = MAX_PROVIDER_FILES if max_files is None else max_files
    idx = ProviderIndex()
    imports = {}
    paths = sorted(p for p in (files or {}) if p != exclude)
    if len(paths) > max_files:
        idx.truncated = True
        paths = paths[:max_files]
    for path in paths:
        try:
            tree = ast.parse(files[path])
        except (SyntaxError, ValueError, RecursionError):
            continue
        idx.files_scanned += 1
        seen_exc, seen_handler = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.level:
                    continue          # relative: not portable to another package
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    binding = _import_binding(node, alias)
                    stmt = _import_stmt_text(node, alias)
                    imports.setdefault(binding, {})
                    imports[binding][stmt] = imports[binding].get(stmt, 0) + 1
            elif isinstance(node, (ast.ExceptHandler, ast.Raise)):
                for name in _exc_names(node):
                    seen_exc.add(name)
                if isinstance(node, ast.ExceptHandler):
                    shape = _handler_shape(node)
                    if shape:
                        seen_handler.add(shape)
        for name in seen_exc:
            idx.exceptions[name] = idx.exceptions.get(name, 0) + 1
        for shape in seen_handler:
            idx.handlers[shape] = idx.handlers.get(shape, 0) + 1
    for binding, forms in imports.items():
        ranked = sorted(forms.items(), key=lambda kv: (-kv[1], kv[0]))
        idx.imports[binding] = [(freq, stmt) for stmt, freq
                                in ranked[:MAX_PROVIDERS_PER_NAME]]
    return idx


def _file_local_index(file_lines):
    """The degenerate index: the parent file itself.

    It can never supply an import for an UNDEFINED name (an import in the file
    binds the name, so the name is not undefined), which is exactly why the
    corpus channel exists. It is still the right source for exception classes
    and handler conventions, which is why the fallback is not simply "empty".
    """
    src = "\n".join(file_lines or [])
    if not src.strip():
        return ProviderIndex()
    return build_provider_index({"<parent>": src})


# ---------------------------------------------------------- scope analysis
def _scope_bindings(fn):
    """Every name the FUNCTION binds, by Python's own binding constructs."""
    names = set()
    for a in list(getattr(fn.args, "posonlyargs", [])) + list(fn.args.args) \
            + list(fn.args.kwonlyargs):
        names.add(a.arg)
    for extra in (fn.args.vararg, fn.args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(_import_binding(node, alias))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not fn:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _module_bindings(file_lines):
    """Names bound at MODULE level in the parent file (top-level only).

    Only the top level: a name bound inside some other function is not visible
    here, and treating it as bound is how an enumerator convinces itself a
    missing import is present.
    """
    names = set()
    if not file_lines:
        return names
    try:
        tree = ast.parse("\n".join(file_lines))
    except (SyntaxError, ValueError):
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    names.add(_import_binding(node, alias))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(node, ast.Try):
            # `try: import x / except ImportError: x = None` is an import block
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        if alias.name != "*":
                            names.add(_import_binding(sub, alias))
    return names


def _undefined_loads(fn, file_lines):
    """Names LOADED in `fn` that nothing in reach binds. Source order, deduped.

    Deliberately conservative: a name bound ANYWHERE in the function counts as
    bound (no flow sensitivity), because a false "undefined" would propose an
    import for a name the function already owns — a candidate that can only be
    wrong, and #163's fail-plausible shape.
    """
    bound = _scope_bindings(fn) | _module_bindings(file_lines) | _BUILTIN_NAMES
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id not in bound and node.id not in out:
            out.append(node.id)
    return out


def _body_anchor(fn, span_lines):
    """``(index, line)`` of the first real statement of `fn`'s body.

    An insertion anchor must be a whole statement at the body's own indentation
    and must never be the docstring, so a function whose
    body is only a docstring has NO anchor and yields no candidate.
    """
    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                and isinstance(st.value.value, str):
            continue
        i = st.lineno - 1
        if 0 <= i < len(span_lines):
            return i, span_lines[i]
        return None
    return None


def _insert_before(span_lines, line_idx, new_stmt, role):
    """A `SpanEdit` that inserts `new_stmt` immediately before `line_idx`.

    The anchor is the whole ANCHOR LINE, so `apply_span_edits`'s uniqueness rule
    still decides: an ambiguous anchor is a rejection, never a guess.
    """
    anchor = span_lines[line_idx]
    if not anchor.strip():
        return None
    return SpanEdit(anchor, "%s\n%s" % (new_stmt, anchor), role=role)


# ----------------------------------------------------- c1: import_change
def c1_import_change(span_lines, start_idx, file_lines=None, providers=None):
    """The `import_change` family: add / remove / retarget an import.

    Three statically grounded signals, in own order:

      1. ADD — a name is LOADED and nothing binds it. The provider comes from
         the corpus index, which is the only place it CAN come from (the parent
         file cannot supply an import for a name the parent file lacks).
      2. REMOVE — a function-local import whose binding is never loaded.
      3. RETARGET — a function-local `from M import N` where the corpus is
         dominated by a DIFFERENT module for the same binding. This is the
         literal census wording, "reaches for the wrong dependency".
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    providers = providers if providers is not None else ProviderIndex()
    out = []

    # ---- 1. ADD a missing import, as a function-local statement
    anchor = _body_anchor(fn, span_lines)
    if anchor is not None:
        anchor_i, anchor_line = anchor
        indent = _indent_of(anchor_line)
        for name in _undefined_loads(fn, file_lines)[:MAX_UNDEFINED_NAMES]:
            for stmt in providers.providers_for(name):
                edit = _insert_before(span_lines, anchor_i, indent + stmt,
                                      "add_import")
                if edit is None:
                    continue
                es = _mk_set(span_lines, start_idx, [edit], "import_add", [name])
                if es is not None:
                    out.append(es)

    # ---- 2. REMOVE an unused function-local import
    loads = {n.id for n in ast.walk(fn)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    local_imports = [n for n in ast.walk(fn)
                     if isinstance(n, (ast.Import, ast.ImportFrom)) and n is not fn]
    for node in sorted(local_imports, key=lambda n: n.lineno):
        bindings = [_import_binding(node, a) for a in node.names if a.name != "*"]
        if not bindings or any(b in loads or b in attrs for b in bindings):
            continue
        i = node.lineno - 1
        if not (0 <= i < len(span_lines)):
            continue
        if getattr(node, "end_lineno", node.lineno) - 1 != i:
            continue                       # multi-line import: not a line delete
        line = span_lines[i]
        if "\n".join(span_lines).count(line + "\n") != 1:
            continue
        edit = SpanEdit(line + "\n", "", role="remove_import")
        es = _mk_set(span_lines, start_idx, [edit], "import_remove", bindings)
        if es is not None:
            out.append(es)

    # ---- 3. RETARGET a local import to the corpus-dominant provider
    for node in sorted(local_imports, key=lambda n: n.lineno):
        if getattr(node, "end_lineno", node.lineno) != node.lineno:
            continue
        if len(node.names) != 1 or node.names[0].name == "*":
            continue
        binding = _import_binding(node, node.names[0])
        current = _import_stmt_text(node, node.names[0])
        i = node.lineno - 1
        if not (0 <= i < len(span_lines)):
            continue
        line = span_lines[i]
        if line.strip() != current:
            continue                       # formatting we did not author: leave it
        for stmt in providers.providers_for(binding):
            if stmt == current:
                continue
            edit = SpanEdit(line, _indent_of(line) + stmt, role="retarget_import")
            es = _mk_set(span_lines, start_idx, [edit], "import_retarget",
                         [binding])
            if es is not None:
                out.append(es)
    return out


# ------------------------------------------------- c2: statement_insert
#: finite summary registry. NAME GUESSING IS DELIBERATELY ABSENT: an
#: unknown API produces no candidate, which is why an empty menu here is a real
#: negative rather than a shrug.
TRANSACTION_WRITES = frozenset(("add", "add_all", "delete", "execute", "flush",
                                "merge", "bulk_save_objects"))
TRANSACTION_RESOLVES = frozenset(("commit", "rollback"))
RESOURCE_ACQUIRE = {"open": "close", "SessionLocal": "close",
                    "acquire": "release", "connect": "close",
                    "raw_connection": "close"}


def _receiver_of(call):
    """The NAME the method is called on, for ``recv.method(...)``."""
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f.value.id, f.attr
    return None, None


def _outer_returns(fn, span_lines):
    """`Return` statements in `fn`'s own statement lists, source order."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and 0 <= node.lineno - 1 < len(span_lines):
            out.append(node)
    return sorted(out, key=lambda n: n.lineno)


def c2_statement_insert(span_lines, start_idx, file_lines=None, providers=None):
    """The `statement_insert` family: one statement, at a legal boundary.

    four signal classes, each capped at `MAX_BOUNDARIES_PER_CLASS`:

      X a recognised transaction write reaches an exit unresolved -> `commit`
         before a `return`, `rollback` before a `raise` in a handler;
      R a registry-recognised resource is acquired and never released ->
         `close()`/`release()` before the last exit;
      U a local is USED before anything binds it -> bind it from the RHS the
         function itself uses later (a donor, never invented text);
      P a peer function in the SAME FILE with the same call shape performs a
         step this one skips -> transfer that statement.

    Every candidate is a statement the file or its own later code already
    contains. Nothing here authors new prose, which is also the ceiling: a gold
    that invents a name is out of reach BY CONSTRUCTION, not by pruning.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    out = []
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]

    # ---- X: unresolved transaction
    writes, resolves = {}, set()
    for call in calls:
        recv, meth = _receiver_of(call)
        if recv is None:
            continue
        if meth in TRANSACTION_WRITES:
            writes.setdefault(recv, call.lineno)
        elif meth in TRANSACTION_RESOLVES:
            resolves.add(recv)
    dirty = sorted(r for r in writes if r not in resolves)
    for recv in dirty[:MAX_BOUNDARIES_PER_CLASS]:
        for ret in _outer_returns(fn, span_lines)[:MAX_BOUNDARIES_PER_CLASS]:
            i = ret.lineno - 1
            if i <= writes[recv] - 1:
                continue                   # the write has not happened yet
            edit = _insert_before(span_lines, i,
                                  "%s%s.commit()" % (_indent_of(span_lines[i]),
                                                     recv),
                                  "insert_commit")
            if edit is None:
                continue
            es = _mk_set(span_lines, start_idx, [edit], "stmt_commit", [recv])
            if es is not None:
                out.append(es)
                break
    for handler in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        bare = [s for s in handler.body
                if isinstance(s, ast.Raise) and s.exc is None]
        if not bare:
            continue
        for recv in dirty[:MAX_BOUNDARIES_PER_CLASS]:
            i = bare[0].lineno - 1
            if not (0 <= i < len(span_lines)):
                continue
            edit = _insert_before(span_lines, i,
                                  "%s%s.rollback()" % (_indent_of(span_lines[i]),
                                                       recv),
                                  "insert_rollback")
            if edit is None:
                continue
            es = _mk_set(span_lines, start_idx, [edit], "stmt_rollback", [recv])
            if es is not None:
                out.append(es)

    # ---- R: unreleased resource
    with_bound = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            with_bound.add(n.id)
    held = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            continue
        f = node.value.func
        key = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)
        if key not in RESOURCE_ACQUIRE:
            continue
        name = node.targets[0].id
        if name in with_bound:
            continue                       # auto-released
        rel = RESOURCE_ACQUIRE[key]
        calls_on_it = {m for r, m in (_receiver_of(c) for c in calls) if r == name}
        if rel in calls_on_it:
            continue
        held.append((node.lineno, name, rel))
    for lineno, name, rel in sorted(held)[:MAX_BOUNDARIES_PER_CLASS]:
        rets = [r for r in _outer_returns(fn, span_lines) if r.lineno > lineno]
        if not rets:
            continue
        i = rets[-1].lineno - 1
        edit = _insert_before(span_lines, i,
                              "%s%s.%s()" % (_indent_of(span_lines[i]), name, rel),
                              "insert_release")
        if edit is None:
            continue
        es = _mk_set(span_lines, start_idx, [edit], "stmt_release", [name, rel])
        if es is not None:
            out.append(es)

    # ---- U: used before bound (donor RHS from the function's own later binding)
    first_load, first_store = {}, {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Name):
            continue
        if isinstance(node.ctx, ast.Load):
            first_load.setdefault(node.id, node.lineno)
        elif isinstance(node.ctx, ast.Store):
            first_store.setdefault(node.id, node.lineno)
    params = {a.arg for a in list(getattr(fn.args, "posonlyargs", []))
              + list(fn.args.args) + list(fn.args.kwonlyargs)}
    early = sorted(n for n in first_load
                   if n in first_store and n not in params
                   and first_load[n] < first_store[n])
    for name in early[:MAX_BOUNDARIES_PER_CLASS]:
        donor = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id == name \
                    and node.lineno == first_store[name]:
                donor = node
                break
        if donor is None:
            continue
        rhs = _unparse(donor.value)
        i = first_load[name] - 1
        if rhs is None or not (0 <= i < len(span_lines)):
            continue
        # the donor's own free names must already be available at the boundary
        rhs_names = {n.id for n in ast.walk(donor.value) if isinstance(n, ast.Name)}
        avail = params | {n for n, ln in first_store.items() if ln < first_load[name]}
        if not rhs_names <= (avail | _BUILTIN_NAMES | _module_bindings(file_lines)):
            continue
        edit = _insert_before(span_lines, i,
                              "%s%s = %s" % (_indent_of(span_lines[i]), name, rhs),
                              "insert_binding")
        if edit is None:
            continue
        es = _mk_set(span_lines, start_idx, [edit], "stmt_bind", [name])
        if es is not None:
            out.append(es)

    # ---- P: peer-function parallelism, mined from the PARENT FILE only
    for stmt_text, coupling in _peer_gap_donors(fn, span_lines, file_lines):
        anchor = _body_anchor(fn, span_lines)
        if anchor is None:
            break
        i, line = anchor
        edit = _insert_before(span_lines, i, _indent_of(line) + stmt_text,
                              "insert_peer_step")
        if edit is None:
            continue
        es = _mk_set(span_lines, start_idx, [edit], "stmt_peer", coupling)
        if es is not None:
            out.append(es)
    return out


def _peer_gap_donors(fn, span_lines, file_lines, max_donors=2):
    """Statements every PEER of `fn` performs and `fn` does not.

    A peer is a module-level function in the same file that calls the same
    external symbol `fn` calls. The donor must be a single expression-statement
    call whose every free name is already bound in `fn`; anything else would
    need alpha-renaming an unbound local, which is how a "transfer" quietly
    invents a name.

    the design mines SIBLING FILES for this. It is restricted to the parent
    file on purpose: a cross-file donor's receiver is a local of a function we
    are not editing, and re-binding it is exactly the invented-name failure the
    reachability report names as the wall. Widening this is the next build, not
    a silent capability.
    """
    if not file_lines:
        return []
    try:
        tree = ast.parse("\n".join(file_lines))
    except (SyntaxError, ValueError):
        return []
    my_calls = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call):
            r, m = _receiver_of(c)
            if m:
                my_calls.add((r, m))
    my_bound = _scope_bindings(fn)
    mine = {m for _r, m in my_calls}
    out = []
    for peer in tree.body:
        if not isinstance(peer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if peer.name == fn.name:
            continue
        peer_calls = {}
        for c in ast.walk(peer):
            if isinstance(c, ast.Call):
                r, m = _receiver_of(c)
                if m:
                    peer_calls.setdefault(m, []).append(c)
        if not (mine & set(peer_calls)):
            continue                       # not a peer: no shared call symbol
        for stmt in ast.walk(peer):
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            r, m = _receiver_of(stmt.value)
            if r is None or m in mine or (r, m) in my_calls:
                continue
            free = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            if not free <= my_bound:
                continue                   # would need an invented binding
            text = _unparse(stmt.value)
            if text is None or (text, (r, m)) in out:
                continue
            out.append((text, [r, m]))
            if len(out) >= max_donors:
                return [(t, list(c)) for t, c in out]
    return [(t, list(c)) for t, c in out]


# --------------------------------------------- c3: exception_handler_change
#: template vocabulary, plus the eligibility rule that keeps a swallow out
#: of the menu unless the FILE itself demonstrates one. `reraise` is the
#: universally safe template and is always eligible.
_HANDLER_BODIES = {
    "reraise": "raise",
    "log_reraise": None,                   # rendered from the file's logger
    "return_none": "return None",
    "pass": "pass",
    "continue": "continue",
}
_BEHAVIOR_CHANGING = frozenset(("return_none", "pass", "continue"))


def _logger_expr(fn, file_lines):
    """The logger expression visible at the target site, or None.

    Never invents a binding: it must be a module-level name or `self.logger`
    the file already uses in a handler.
    """
    mod = _module_bindings(file_lines)
    for cand in ("logger", "log", "LOGGER"):
        if cand in mod:
            return cand
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "logger" \
                and isinstance(node.value, ast.Name) and node.value.id == "self":
            return "self.logger"
    return None


def _risk_types(node, file_lines, providers):
    """Exception classes plausibly raised under `node`, best evidence first.

    Evidence, in priority order and NOTHING beyond it:
      1. classes the PARENT FILE itself raises or catches (usage corpus);
      2. classes the provider corpus names, ranked by support.
    A class the target scope cannot resolve is rejected, because an unresolvable
    name is a NameError wearing a fix's clothes.
    """
    if not any(isinstance(n, ast.Call) for n in ast.walk(node)):
        return []
    visible = _module_bindings(file_lines) | _BUILTIN_NAMES
    local, ranked = {}, []
    try:
        tree = ast.parse("\n".join(file_lines or []))
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        for n in ast.walk(tree):
            for name in _exc_names(n):
                local[name] = local.get(name, 0) + 1
    for name, freq in sorted(local.items(), key=lambda kv: (-kv[1], kv[0])):
        if name in visible and name not in ranked:
            ranked.append(name)
    corpus = (providers.exceptions if providers is not None else {})
    for name, _freq in sorted(corpus.items(), key=lambda kv: (-kv[1], kv[0])):
        if name in visible and name not in ranked:
            ranked.append(name)
    if "Exception" in visible and "Exception" not in ranked:
        ranked.append("Exception")         # generic LAST, never first
    return ranked[:MAX_EXC_TYPES_PER_SITE]


def _eligible_templates(fn, file_lines, providers, in_loop=False,
                        local_handlers=None, for_wrap=False):
    """Handler templates this site may use, safest first.

    ``for_wrap`` drops ``reraise``. ``try: S / except X: raise`` is EXACTLY
    equivalent to ``S`` — it is not a weak candidate, it is a candidate that
    cannot be a fix under any scorer, and on the v3 census tasks it was roughly
    a third of the wrap enumeration. A menu slot spent on a provable no-op is
    worse than an empty slot, because it looks like coverage.

    ``local_handlers`` is the parent file's own handler-shape counts, hoisted by
    the caller: recomputing it inside the wrap loop re-parsed the whole file
    once per (site x type x template).
    """
    out = [] if for_wrap else ["reraise"]
    logger = _logger_expr(fn, file_lines)
    if logger:
        out.append("log_reraise")
    local = (_file_local_index(file_lines).handlers if local_handlers is None
             else local_handlers)
    for name in ("return_none", "pass", "continue"):
        if name == "continue" and not in_loop:
            continue
        if local.get(name):                # the FILE demonstrates this shape
            out.append(name)
    return out


def _render_handler(indent, exc, template, fn, file_lines):
    logger = _logger_expr(fn, file_lines)
    if template == "log_reraise":
        if not logger:
            return None
        body = ['%s    %s.exception("%s")' % (indent, logger, "unhandled error"),
                "%s    raise" % indent]
    else:
        text = _HANDLER_BODIES.get(template)
        if not text:
            return None
        body = ["%s    %s" % (indent, text)]
    return ["%sexcept %s:" % (indent, exc)] + body


def c3_exception_handler(span_lines, start_idx, file_lines=None, providers=None):
    """The `exception_handler_change` family: wrap / add-clause / narrow.

    Operator A wraps a contiguous run of risky statements in `try/except`,
    operator B adds a missing clause to an existing `try`, operator C narrows a
    bare `except`. Behaviour-changing templates need the parent file's own
    evidence, and `BaseException` is never a candidate.
    """
    tree, _ = _parse_span(span_lines)
    if tree is None:
        return []
    fn = _first_function(tree)
    if fn is None:
        return []
    providers = providers if providers is not None else ProviderIndex()
    out = []
    # Hoisted out of the wrap loop below: both of these parse the WHOLE parent
    # file, and inside the loop they ran once per (site x type x template).
    local_handlers = _file_local_index(file_lines).handlers
    loops = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            for sub in ast.walk(node):
                loops.add(id(sub))

    # ---- C: narrow a bare `except:`
    for handler in sorted([n for n in ast.walk(fn)
                           if isinstance(n, ast.ExceptHandler) and n.type is None],
                          key=lambda n: n.lineno):
        i = handler.lineno - 1
        if not (0 <= i < len(span_lines)):
            continue
        line = span_lines[i]
        if line.strip() != "except:":
            continue
        owner = None
        for t in ast.walk(fn):
            if isinstance(t, ast.Try) and handler in t.handlers:
                owner = t
                break
        if owner is None:
            continue
        covered = set()
        for h in owner.handlers:
            if h is handler:
                break
            covered.update(_exc_names(h))
        for exc in _risk_types(owner, file_lines, providers):
            if exc in covered:
                continue
            edit = SpanEdit(line, "%sexcept %s:" % (_indent_of(line), exc),
                            role="narrow_bare_except")
            es = _mk_set(span_lines, start_idx, [edit], "except_narrow", [exc])
            if es is not None:
                out.append(es)

    # ---- B: add a missing clause to an existing `try`
    for node in sorted([n for n in ast.walk(fn) if isinstance(n, ast.Try)],
                       key=lambda n: n.lineno):
        if not node.handlers:
            continue
        covered = set()
        for h in node.handlers:
            if h.type is None:
                covered.add("*")
            covered.update(_exc_names(h))
        if "*" in covered:
            continue                       # a bare handler makes any add dead
        first = node.handlers[0]
        i = first.lineno - 1
        if not (0 <= i < len(span_lines)):
            continue
        indent = _indent_of(span_lines[i])
        in_loop = id(node) in loops
        for exc in _risk_types(node, file_lines, providers):
            if exc in covered or exc == "Exception":
                continue                   # a generic ADD shadows every later one
            for template in _eligible_templates(fn, file_lines, providers,
                                                in_loop,
                                                local_handlers=local_handlers):
                block = _render_handler(indent, exc, template, fn, file_lines)
                if block is None:
                    continue
                edit = _insert_before(span_lines, i, "\n".join(block),
                                      "add_except_clause")
                if edit is None:
                    continue
                es = _mk_set(span_lines, start_idx, [edit], "except_add", [exc])
                if es is not None:
                    out.append(es)

    # ---- A: wrap a contiguous run of risky statements
    sites = 0
    for body in _stmt_lists(fn):
        if sites >= MAX_WRAP_SITES:
            break
        for start in range(len(body)):
            if sites >= MAX_WRAP_SITES:
                break
            for width in range(1, MAX_WRAP_WIDTH + 1):
                end = start + width - 1
                if end >= len(body):
                    break
                span = body[start:end + 1]
                if any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef, ast.Return, ast.Try))
                       for s in span):
                    continue
                lo = span[0].lineno - 1
                hi = getattr(span[-1], "end_lineno", span[-1].lineno) - 1
                if not (0 <= lo <= hi < len(span_lines)):
                    continue
                types = _risk_types(span[0] if width == 1 else ast.Module(
                    body=list(span), type_ignores=[]), file_lines, providers)
                if not types:
                    continue
                indent = _indent_of(span_lines[lo])
                if any(_indent_of(span_lines[k]) != indent
                       for k in range(lo, hi + 1) if span_lines[k].strip()):
                    pass                   # continuation lines are fine
                old = "\n".join(span_lines[lo:hi + 1])
                if "\n".join(span_lines).count(old) != 1:
                    continue
                sites += 1
                in_loop = any(id(s) in loops for s in span)
                templates = _eligible_templates(
                    fn, file_lines, providers, in_loop,
                    local_handlers=local_handlers, for_wrap=True)
                for exc in types:
                    for template in templates:
                        handler = _render_handler(indent, exc, template, fn,
                                                  file_lines)
                        if handler is None:
                            continue
                        indented = "\n".join(
                            ("    " + l) if l.strip() else l
                            for l in span_lines[lo:hi + 1])
                        new = "\n".join(["%stry:" % indent, indented] + handler)
                        edit = SpanEdit(old, new, role="wrap_try")
                        es = _mk_set(span_lines, start_idx, [edit],
                                     "except_wrap", [exc])
                        if es is not None:
                            out.append(es)
                break                      # one width per start: the narrowest
    return out


#: The census tranche. Appended after `GENERATORS` and drawn from its OWN
#: budget, so a coordination candidate can never be evicted or reordered by it.
CENSUS_GENERATORS = (
    ("import_change", c1_import_change),
    ("statement_insert", c2_statement_insert),
    ("exception_handler", c3_exception_handler),
)


#: THE CENSUS PUNTS — declared, with the token that blocks each one, measured
#: against the parent file AND the 3,519-file repo corpus by
#: ``probes/census_reachability.py``. Same two classes `PUNTED_PATTERNS` uses.
#:
#: MEASURED RESULT: 0 of the 6 v3 census tasks is gold-reachable, in EITHER
#: configuration. That is not a pruning failure and more search will not move
#: it. Every one of the six golds contains at least one token that exists
#: nowhere the enumerator can see, and AST equality counts every token.
#:
#: The corpus channel is still doing real work: it shrinks the blocking set from
#: 11 tokens to 1 on `_runner_ws_uuid`, from 3 to 1 on `override_mr_checks`, and
#: moves two tasks from NOVEL-TOKEN to ASSEMBLY outright. What it cannot supply
#: is prose a human authored for this MR.
CENSUS_PUNTS = {
    "import_add_with_its_own_first_use": (
        "NOVEL-TOKEN. v3 tasks 2dbb4371 / 7787d96f / a9af5404, all three census "
        "`import_change` rows. `c1_import_change` fires on an UNDEFINED NAME — a "
        "name the function loads and nothing binds. In all three golds the MR "
        "adds the import AND its first use in the same hunk, so the buggy "
        "function has no undefined name and there is no signal to fire on. "
        "Measured: `_undefined_loads` returns [] for all three. Reaching these "
        "needs a generator that authors the USE, which is a behaviour proposal, "
        "not an import fix."),
    "authored_message_string": (
        "NOVEL-TOKEN. v3 tasks 2dbb4371 ('Workspace header disagrees with task "
        "grant') and 7787d96f ('Overriding a required check is a human "
        "action...'). Each string occurs in exactly ONE file of the 3,519-file "
        "corpus — the fixed file itself. A hard wall for any enumerator: the "
        "text did not exist until a human wrote it."),
    "invented_local_binding": (
        "NOVEL-TOKEN/ASSEMBLY. v3 tasks fb1d75cb (`_SAFETY_RELEVANT_FIELDS`, "
        "`merged`) and a9af5404 (`incident_db`). `_SAFETY_RELEVANT_FIELDS` is "
        "corpus-unique -> NOVEL-TOKEN. `incident_db` occurs in 4 corpus files, "
        "so a9af5404 is ASSEMBLY: every token exists, and composing the "
        "session-isolation shape (bind a new session, re-thread it through the "
        "call, close it) is the next real build order — the same class the "
        "coordination family's `branch_specialized_fallback` was in before it "
        "landed."),
    "unreleased_symbol_from_the_same_mr": (
        "NOVEL-TOKEN. v3 task 09523282, the only `exception_handler_change`. "
        "The gold catches `PolicyUnavailable`, which occurs in ZERO of the 3,519 "
        "corpus files: the MR introduces the exception class and the handler "
        "together. `_risk_types` deliberately refuses to name a class the target "
        "scope cannot resolve, so this is the enumerator being correct, not the "
        "enumerator being narrow."),
    "cross_file_donor_transfer": (
        "ASSEMBLY. v3 task 252c12de. The gold inserts "
        "`_rl.record_proposal(...)` before `_rl.record_verdict(...)`; the donor "
        "lives in a SIBLING FILE (routers/helm_decisions.py) in a different "
        "function, with its receiver bound to that function's own local "
        "`decision`. `_peer_gap_donors` mines the PARENT FILE only, precisely "
        "because transferring a cross-file donor means re-binding a local we "
        "cannot see — which is how a 'transfer' quietly invents a name. The proposed "
        "design assumes this channel; widening to it is the next build, and it "
        "needs an argument-rebinding gate before it is safe."),
}


# ============================================================ 5. ENTRY POINT
def enumerate_edit_sets(span_lines, start_idx, file_lines=None, cap=None,
                        providers=None):
    """The coordinated family's closed candidate set.

    Deterministic: generators run in a fixed order, each is itself deterministic
    (sorted iteration over AST facts), and the result is deduped stably. Returns
    at most `MAX_EDIT_SETS` coordination candidates followed by at most
    `MAX_CENSUS_SETS` census candidates; `search_exhausted` reports whether a
    hard exploration cap was hit, so a truncation is never silent.
    """
    return enumerate_with_metadata(span_lines, start_idx, file_lines, cap,
                                   providers)[0]


def _run_tranche(generators, span_lines, start_idx, file_lines, providers,
                 explore_cap, seen, pass_providers, round_robin=False):
    """One generator tranche under its OWN exploration budget.

    Returns ``(candidates, explored, truncated, failed)``. A generator that
    raises is RECORDED, never swallowed into silence (law 1) — a crashed
    generator is the difference between "no fix here" and "we did not look".

    ``round_robin`` takes each generator's FIRST candidate before any
    generator's second — the same fairness fix `fixer.build_mutation_menu` and
    `mutations.enumerate_mutations` already carry, for the same reason:
    concatenation order is not a relevance order, so the first generator with a
    big output eats the head of a capped pool and every later generator's BEST
    candidate ranks below its WORST.

    It is OFF for the coordination tranche (whose order is measured and must not
    move) and ON for the census tranche, where `c3_exception_handler` produces
    31-72 candidates against `c1`'s 0-1 and `c2`'s 0-2 on the six v3 census
    tasks. HONESTY NOTE: on those six it changes nothing measurable — `c3` runs
    LAST, so plain concatenation already happens to spare the small generators.
    Round-robin is here as a STRUCTURAL guarantee that survives a generator
    being added, reordered, or growing, not as a measured win on this board.
    The guard that proves it is `test_run_tranche_round_robin_...`, which drives
    `_run_tranche` with the flood FIRST — the ordering the real tranche does not
    currently have and must not silently acquire.
    """
    explored, truncated, failed, produced = 0, False, [], []
    for name, gen in generators:
        try:
            got = (gen(span_lines, start_idx, file_lines, providers)
                   if pass_providers
                   else gen(span_lines, start_idx, file_lines))
        except Exception as exc:             # law 1: rather learn nothing
            failed.append("%s:%s" % (name, type(exc).__name__))
            continue
        produced.append(list(got))

    out = []

    def _take(cand):
        nonlocal explored, truncated
        explored += 1
        if explored > explore_cap:
            truncated = True
            return False
        key = (cand[0], cand[1], cand[2])
        if key not in seen:
            seen.add(key)
            out.append(cand)
        return True

    if round_robin:
        rank = 0
        while not truncated:
            progressed = False
            for cands in produced:
                if rank >= len(cands):
                    continue
                progressed = True
                if not _take(cands[rank]):
                    break
            if not progressed:
                break
            rank += 1
    else:
        for cands in produced:
            for cand in cands:
                if not _take(cand):
                    break
            if truncated:
                break
    return out, explored, truncated, failed


def enumerate_with_metadata(span_lines, start_idx, file_lines=None, cap=None,
                            providers=None):
    """``(candidates, metadata)``.

    S14 from the executor steal-list, ported: an EMPTY candidate list may be read
    as "this function has no coordinated fix" ONLY when the search provably
    completed and was not truncated. ``negative_conclusion_allowed`` carries that
    licence explicitly, so an abstain can never quietly launder a truncated or
    crashed search into a finding.
    """
    cap = MAX_EDIT_SETS if cap is None else cap
    seen = set()
    out, explored, truncated, failed = _run_tranche(
        GENERATORS, span_lines, start_idx, file_lines, providers,
        MAX_EXPLORED_SETS, seen, pass_providers=False)
    if len(out) > cap:
        truncated = True
    out = out[:cap]
    # CENSUS TRANCHE — its own budget, its own truncation flag, appended after
    # the coordination candidates are already final. That is what makes the
    # no-regression claim structural rather than a hope: `out` above is
    # byte-identical to what it was before this tranche existed, and nothing
    # below can reorder or evict it.
    c_out, c_explored, c_trunc, c_failed = _run_tranche(
        CENSUS_GENERATORS, span_lines, start_idx, file_lines, providers,
        MAX_CENSUS_EXPLORED, seen, pass_providers=True, round_robin=True)
    if len(c_out) > MAX_CENSUS_SETS:
        c_trunc = True
    c_out = c_out[:MAX_CENSUS_SETS]
    meta = {
        "emitted": len(out),
        "explored": explored,
        "truncated": truncated,
        "generators_failed": failed,
        # an empty menu means "nothing here" only if nothing was cut short
        "negative_conclusion_allowed": (not truncated) and not failed,
        # The census tranche reports SEPARATELY. Folding it into the flags above
        # would silently revoke the coordination family's negative-conclusion
        # licence the first time a census generator truncated — a claim about
        # one search answered with another search's bookkeeping.
        "census_emitted": len(c_out),
        "census_explored": c_explored,
        "census_truncated": c_trunc,
        "census_generators_failed": c_failed,
        "census_negative_conclusion_allowed": (not c_trunc) and not c_failed,
        "providers": None if providers is None else {
            "files_scanned": providers.files_scanned,
            "truncated": providers.truncated,
            "bindings": len(providers.imports),
        },
    }
    return out + c_out, meta


def search_exhausted(span_lines, start_idx, file_lines=None, providers=None):
    """True when a hard exploration cap bounded EITHER tranche for this span."""
    meta = enumerate_with_metadata(span_lines, start_idx, file_lines,
                                   providers=providers)[1]
    return meta["truncated"] or meta["census_truncated"]
