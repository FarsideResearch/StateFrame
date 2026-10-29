"""TYPED HOLES — the novel-token protocol.

Why this file exists
--------------------
MEASURED, not asserted: 16 of the 21 real v3 golds contain at least one token
that occurs nowhere in the parent file — invented locals (``_cc``,
``_claimed_author``, ``_SAFETY_RELEVANT_FIELDS``), authored prose
(``'Turn dispatch failed'``), corpus-unique class names. ``edit_sets``'
`PUNTED_PATTERNS` and `CENSUS_PUNTS` classify every one of those misses as
NOVEL-TOKEN, and that classification is exactly right: an enumerator over the
parent file can only ever re-propose tokens the parent file already contains.
More search does not move it. It is a hard expressiveness boundary.

The protocol that crosses it, and its safe boundary
---------------------------------------------------
    The host chooses WHERE code may change. The model chooses only the bytes
    occupying ONE TYPED HOLE. Everything else is deterministic validation.

So enumeration still produces the whole structure, deterministically, from the
parent alone — but a `HoleTemplate` leaves one or more typed slots empty. The
model fills a slot with a short, grammar-shaped, no-think completion; the fill
is validated deterministically (parse, identifier rules, scope collision, style)
and then INSTANTIATED into an ordinary `edit_sets.EditSet` that enters the
normal pipeline. A fill never bypasses arbitration: it competes in spec_exec,
witness and ballot exactly like an enumerated candidate, because by the time
arbitration sees it, it IS one.

Three things this file refuses to do
------------------------------------
1. **It never lets the model emit a patch.** The model sees a slot description
   and context and returns one value. It never supplies a source range, an AST
   path, a replacement count, or a line of code. That is what makes the blast
   radius of a bad fill "one token in a pre-validated structure" rather than
   "arbitrary rewrite".
2. **It never relaxes the strict metric.** A behaviourally-right,
   token-different fill is recorded as `near_miss_token_only` on the cert and
   reported SEPARATELY. `exact` stays exact. Collapsing the two would be the
   one number that makes the bench useless.
3. **It never claims template coverage it has not measured.** `oracle_fill`
   exists precisely so the bench can answer "how many novel-token golds would we
   reach if the model were PERFECT" — isolating template coverage from fill
   quality. See ``probes/hole_reachability.py`` for the table, which is reported
   honestly including the rows this protocol does not reach.

Grammar note (w35)
------------------
Grammar-constrained decoding and thinking are incompatible in current
llama.cpp builds (grammar enforcement silently goes inactive when thinking is
on). So the fill call is a SEPARATE, short, no-think completion, and the
grammar supplements the post-validator: `validate_fill` is authoritative
and runs whether or not a grammar was applied. This mirrors the `grammar_pick`
lesson already recorded in `gym_config`.
"""

from __future__ import annotations

import ast
import hashlib
import keyword
import re

import edit_sets as es

# ------------------------------------------------------------------ constants

#: Slot kinds. The kind determines the grammar, the validator, and the
#: serialization policy — never the model.
KIND_IDENTIFIER = "identifier"
KIND_STRING = "string_value"
KIND_KWARG = "kwarg_name"
KIND_ATTRIBUTE = "attribute_name"
#: A TEXT RUN INSIDE AN F-STRING. Deliberately its own kind rather than a
#: special case of `string_value`, because its serialization policy is
#: genuinely different: it carries no quotes of its own, `{` and `}` must be
#: doubled, and the interpolations around it are structure the model may not
#: touch. Treating it as a plain literal is exactly the bug the oracle probe
#: caught on v3 row 9 — see `standalone_strings`.
KIND_FSTRING_TEXT = "fstring_text"
KINDS = (KIND_IDENTIFIER, KIND_STRING, KIND_KWARG, KIND_ATTRIBUTE,
         KIND_FSTRING_TEXT)

#: Slot roles — what the fill DOES, which is what the scope validator checks.
ROLE_NEW_BINDING = "new_binding"
ROLE_BINDING_AND_REFERENCES = "binding_and_references"
ROLE_REFERENCE = "reference"
ROLE_MESSAGE = "message"
ROLE_ATTRIBUTE = "attribute"
ROLE_KWARG = "kwarg"

#: The placeholder token. It must be a legal Python identifier (so a template
#: PARSES and can run every `_mk_set` gate before any model is involved) and it
#: must be legal inside a string literal (so a string hole uses the same token).
#: A parent file that already contains this prefix disqualifies the whole span —
#: see `_placeholder_safe`. That is a fail-closed check, not a rename dance.
PLACEHOLDER_PREFIX = "__hole"


def placeholder(index):
    """The sentinel occupying hole ``index``. Deterministic, identifier-legal."""
    return "%s%d__" % (PLACEHOLDER_PREFIX, index)


_PLACEHOLDER_RE = re.compile(r"%s\d+__" % re.escape(PLACEHOLDER_PREFIX))

# ------------------------------------------------------------------ honest caps
#
# Task budgets bound execution cost and
# REPORTED, not to make the search narrow. These are generous, and every one of
# them is surfaced in the cert telemetry so a truncation is never silent.

#: Templates emitted per function, across every template generator.
MAX_HOLE_TEMPLATES = 12
#: Templates a generator may EXPLORE per function before it stops. Hard.
MAX_HOLE_EXPLORED = 96
#: Typed holes a STRUCTURAL template may carry. More than this and the model is
#: writing the patch, which is the thing this protocol exists to prevent.
MAX_HOLES_PER_TEMPLATE = 3
#: Holes a PURE-SUBSTITUTION template may carry. Higher, and the distinction is
#: principled rather than convenient: a substitution template changes NO
#: structure at all (the instantiated AST is the parent's AST with leaves
#: swapped), so every extra hole widens what the model may spell and not what it
#: may build. MEASURED: v3 row 9 (`_compose_test`) rewrites seven authored
#: prompt literals in one commit and nothing else — capping it at 3 would report
#: "out of reach" about a gold that is a pure leaf swap.
#: `_finish` enforces whichever ceiling applies, so the structural bound is not
#: weakened by this existing.
MAX_SUBSTITUTION_HOLES = 8
#: Fill calls per task, across every hole of every template. The real spend cap.
MAX_FILL_CALLS = 64
#: Samples requested per hole. Deduped, so N=3 may yield fewer candidates.
FILL_SAMPLES = 3
#: Hard bound on a fill's decoded length, by kind.
MAX_IDENTIFIER_CHARS = 40
#: MEASURED 2026-08-29: at 200 this rejected the ONLY gold whose shape a
#: template covered (v3 row 8, `_env_guidance`), whose authored literals run to
#: ~700 characters of LLM prompt. A length cap that silently excludes the one
#: reachable row is a cap chosen to make the number look tidy. Generous per the
#: step-generosity doctrine; the real bound on a fill is the model's own budget,
#: and an over-long fill is REJECTED and counted, never truncated into a
#: plausible wrong answer.
MAX_STRING_CHARS = 2000

#: Literal holes mined per span by `h_literal_value`. Bounds a prose-heavy file.
MAX_LITERAL_HOLES = 6
#: Base candidates whose literals get hole-variants in `h_candidate_literal`.
MAX_BASE_CANDIDATES = 6

#: Names a fill may never take, regardless of style. `_BUILTIN_NAMES` is
#: deliberately NOT here: shadowing a builtin is a style smell the scorer
#: penalises, not a hard reject, because real code does it on purpose.
FORBIDDEN_NAMES = frozenset(("self", "cls", "__hole", "None", "True", "False"))


# =============================================================== 1. STRUCTURES
#: Names that are legal, plausible, and usually the model giving up. w37's
#: measured failure mode: a 1B under-constrained on an invention hole emits
#: `tmp`/`foo`/`data` rather than a name that says what the binding IS.
#:
#: NOT universally rejected — the design is explicit that `i` is right for a
#: loop index and `tmp` for a genuinely short-lived local. The rule is
#: ROLE-AWARE: a hole whose whole purpose is to NAME something the reader will
#: meet again (a new binding threaded through several references) may not be
#: filled with a name that carries no information. A plain reference may.
GENERIC_NAMES = frozenset((
    "x", "y", "z", "i", "j", "k", "n", "tmp", "temp", "foo", "bar", "baz",
    "data", "value", "val", "obj", "result", "res", "item", "thing", "var",
    "out", "ret", "arg", "args", "kwargs", "stuff", "info",
))

#: Novelty policy per w37 ss.1. The host decides this, never the model.
NOVELTY_REQUIRED = "required"     # the fill MUST be a token the file lacks
NOVELTY_PREFERRED = "preferred"   # reuse is a risk penalty, not a rejection
NOVELTY_ALLOWED = "allowed"       # reuse is fine (a reference hole)


class Hole:
    """One typed slot. Immutable; the model sees a projection of this."""

    __slots__ = ("hole_id", "kind", "role", "index", "occurrences",
                 "forbidden", "style", "context_before", "context_after",
                 "max_chars", "novelty")

    def __init__(self, hole_id, kind, role, index, occurrences=1,
                 forbidden=(), style=None, context_before="",
                 context_after="", max_chars=None, novelty=None):
        if kind not in KINDS:
            raise ValueError("unknown hole kind %r" % (kind,))
        self.hole_id = hole_id
        self.kind = kind
        self.role = role
        self.index = index
        self.occurrences = occurrences
        self.forbidden = frozenset(forbidden)
        self.style = style or {}
        self.context_before = context_before
        self.context_after = context_after
        self.max_chars = max_chars or (
            MAX_STRING_CHARS
            if kind in (KIND_STRING, KIND_FSTRING_TEXT)
            else MAX_IDENTIFIER_CHARS)
        # A hole that EXISTS to invent a name defaults to novelty=required:
        # that is the entire measured reason this protocol was built, and a
        # binding hole filled with a name already in the file is the
        # copy-attractor failure w37 names first.
        self.novelty = novelty or (
            NOVELTY_REQUIRED
            if role in (ROLE_NEW_BINDING, ROLE_BINDING_AND_REFERENCES)
            else NOVELTY_ALLOWED)

    @property
    def token(self):
        return placeholder(self.index)

    def key(self):
        """Stable identity for dedup and telemetry."""
        return (self.hole_id, self.kind, self.role, self.index)

    def __repr__(self):                                   # pragma: no cover
        return "Hole(%s, %s/%s, x%d)" % (self.hole_id, self.kind, self.role,
                                         self.occurrences)


class HoleTemplate:
    """An edit-set candidate whose spans contain typed holes.

    ``edits`` are ordinary `edit_sets.SpanEdit`s that happen to contain
    placeholder tokens. ``skeleton`` is the `EditSet` those edits produce with
    the placeholders still in place — it exists so every `_mk_set` gate
    (coupling, envelope, atomic application, parse) has ALREADY run before a
    model is ever called. A template that cannot become a valid candidate is
    never emitted, so a fill can never be blamed for a broken structure.
    """

    __slots__ = ("template_id", "pattern", "edits", "holes", "skeleton",
                 "provenance", "base_pattern")

    def __init__(self, pattern, edits, holes, skeleton, provenance="",
                 base_pattern=""):
        self.pattern = pattern
        self.edits = tuple(edits)
        self.holes = tuple(holes)
        self.skeleton = skeleton
        self.provenance = provenance
        self.base_pattern = base_pattern
        self.template_id = _template_id(pattern, edits)

    @property
    def n_holes(self):
        return len(self.holes)

    def hole_by_index(self, index):
        for h in self.holes:
            if h.index == index:
                return h
        return None

    def describe(self):
        return "%s: %d hole(s) [%s]" % (
            self.pattern, len(self.holes),
            ", ".join("%s:%s" % (h.hole_id, h.kind) for h in self.holes))

    def __repr__(self):                                   # pragma: no cover
        return "HoleTemplate(%s)" % self.describe()


def _template_id(pattern, edits):
    """Deterministic identity: same structure -> same id, always."""
    h = hashlib.sha256()
    h.update(pattern.encode("utf-8"))
    for ed in edits:
        h.update(b"\x00")
        h.update(ed.old.encode("utf-8"))
        h.update(b"\x01")
        h.update(ed.new.encode("utf-8"))
    return h.hexdigest()[:16]


# ============================================================ 2. INSTANTIATION
def _substitute(text, fills):
    """Replace every placeholder in ``text`` with its fill. Raises on a gap.

    A missing fill is an ERROR, never a silently-surviving placeholder: a
    placeholder that reaches `_mk_set` would produce a candidate that parses,
    applies, and is nonsense — exactly the plausible-wrong-answer failure the
    silent-drop class is made of.
    """
    def repl(m):
        tok = m.group(0)
        if tok not in fills:
            raise KeyError(tok)
        return fills[tok]
    return _PLACEHOLDER_RE.sub(repl, text)


def instantiate(template, fills, span_lines, start_idx):
    """A concrete `edit_sets.EditSet`, or ``None``.

    ``fills`` maps hole INDEX -> decoded value. String fills are serialized by
    the HOST (`_serialize_string`), never by the model, so a model can never
    inject a quote, an escape, an expression, or adjacent code through a string
    slot.

    Returns ``None`` — never a partial or a placeholder-bearing set — whenever
    the fill is invalid or the instantiated set fails any `_mk_set` gate. The
    caller therefore cannot observe a half-applied template.
    """
    token_map = {}
    for hole in template.holes:
        if hole.index not in fills:
            return None
        value = fills[hole.index]
        verdict = validate_fill(hole, value, template)
        if not verdict.ok:
            return None
        # A string hole's placeholder ALREADY sits inside the quotes the
        # template wrote (`"__hole0__"`), so the fill substitutes the escaped
        # BODY. Re-serializing here would emit `""value""` — a syntax error the
        # parse gate catches, which is how this was found: it silently cost the
        # single gold whose shape a template covered.
        if hole.kind == KIND_STRING:
            token_map[hole.token] = _escape_body(value, hole)
        else:
            # identifier / kwarg / attribute / fstring_text all substitute
            # VERBATIM. `fstring_text` is safe to do so because `validate_fill`
            # already refused every character that would need escaping — the
            # validator, not the serializer, is what makes it safe.
            token_map[hole.token] = value
    try:
        edits = [es.SpanEdit(ed.old, _substitute(ed.new, token_map), ed.role)
                 for ed in template.edits]
    except KeyError:
        return None                       # a placeholder with no declared hole
    out = es._mk_set(span_lines, start_idx, edits,
                     template.pattern, _coupling_of(template))
    if out is None:
        return None
    if _PLACEHOLDER_RE.search(out[2]):
        return None                       # Never emit a live placeholder.
    return out


def _coupling_of(template):
    base = list(getattr(template.skeleton, "coupling", ()) or ())
    return base or [template.pattern]


def _escape_body(value, hole):
    """The escaped CONTENT of a string literal — no surrounding quotes.

    Host-owned, always. The model supplies the DECODED value and never sees a
    quote character in its own output contract, which is exactly what stops a
    string slot from being an injection point for adjacent code.
    """
    quote = hole.style.get("quote", '"')
    body = value.replace("\\", "\\\\").replace(quote, "\\" + quote)
    return (body.replace("\n", "\\n").replace("\r", "\\r")
            .replace("\t", "\\t"))


def _serialize_string(value, hole):
    """A complete string literal. Used to write the TEMPLATE's placeholder."""
    quote = hole.style.get("quote", '"')
    return "%s%s%s" % (quote, _escape_body(value, hole), quote)


# ============================================================== 3. VALIDATION
class FillVerdict:
    """Why a fill was accepted or rejected. Rejections are always NAMED.

    Two layers, per w37, and the separation is the point: `ok` is a
    DETERMINISTIC legality verdict, while `copy_risk`, `generic_risk` and
    `style_score` are SOFT features. A soft feature never rejects on its own —
    it ranks and it warns — because reusing a nearby name is sometimes right,
    `i` is sometimes the right loop index, and a paraphrase is sometimes better
    prose than the gold. Letting a heuristic veto a legal fill is how a
    validator starts encoding taste as correctness.
    """

    __slots__ = ("ok", "reason", "style_score", "copy_risk", "generic_risk")

    def __init__(self, ok, reason="", style_score=0.0, copy_risk=0.0,
                 generic_risk=0.0):
        self.ok = ok
        self.reason = reason
        self.style_score = style_score
        self.copy_risk = copy_risk
        self.generic_risk = generic_risk

    def features(self):
        return {"style_score": round(self.style_score, 3),
                "copy_risk": round(self.copy_risk, 3),
                "generic_risk": round(self.generic_risk, 3)}

    def __repr__(self):                                   # pragma: no cover
        return "FillVerdict(%s, %s, style=%.2f, copy=%.2f, generic=%.2f)" % (
            self.ok, self.reason, self.style_score, self.copy_risk,
            self.generic_risk)


REJECT_TYPE = "REJECT_TYPE"
REJECT_EMPTY = "REJECT_EMPTY"
REJECT_LENGTH = "REJECT_LENGTH"
REJECT_GRAMMAR = "REJECT_GRAMMAR"
REJECT_KEYWORD = "REJECT_KEYWORD"
REJECT_FORBIDDEN = "REJECT_FORBIDDEN"
REJECT_COLLISION = "REJECT_COLLISION"
REJECT_PLACEHOLDER = "REJECT_PLACEHOLDER"
REJECT_CONTROL_CHAR = "REJECT_CONTROL_CHAR"
REJECT_STYLE_HARD = "REJECT_STYLE_HARD"
REJECT_PARSE = "REJECT_PARSE"
#: w37's two named 1B failure modes, as HARD rejections only where the hole's
#: own policy makes them one.
REJECT_COPY = "REJECT_COPY"
REJECT_DEGENERATE = "REJECT_DEGENERATE"


def validate_fill(hole, value, template=None):
    """Deterministic, fail-closed validation of ONE fill. Never raises."""
    if not isinstance(value, str):
        return FillVerdict(False, REJECT_TYPE)
    if value == "":
        return FillVerdict(False, REJECT_EMPTY)
    if len(value) > hole.max_chars:
        return FillVerdict(False, REJECT_LENGTH)
    if PLACEHOLDER_PREFIX in value:
        return FillVerdict(False, REJECT_PLACEHOLDER)

    if hole.kind == KIND_FSTRING_TEXT:
        # A brace the model wrote would open an INTERPOLATION — i.e. it would
        # be writing code, in the one place this protocol promises it cannot.
        # Doubling it silently would change the rendered text; rejecting is the
        # only answer that is both safe and honest.
        if "{" in value or "}" in value:
            return FillVerdict(False, REJECT_GRAMMAR)
        if any(ord(c) < 32 for c in value):
            return FillVerdict(False, REJECT_CONTROL_CHAR)
        if '"' in value or "'" in value or "\\" in value:
            return FillVerdict(False, REJECT_GRAMMAR)
        return FillVerdict(True, "", _string_style_score(hole, value))

    if hole.kind == KIND_STRING:
        if any(ord(c) < 32 and c not in "\n\t" for c in value):
            return FillVerdict(False, REJECT_CONTROL_CHAR)
        if "\n" in value and not hole.style.get("allow_newline"):
            return FillVerdict(False, REJECT_CONTROL_CHAR)
        # HARD style gate, and deliberately the ONLY one for prose: a message
        # hole whose neighbours all carry format arguments must carry the same
        # ones. Everything else about prose is SOFT (`style_score`) — forcing a
        # paraphrase to fail would be scoring wording as correctness.
        want = hole.style.get("format_args")
        if want is not None and _format_signature(value) != tuple(want):
            return FillVerdict(False, REJECT_STYLE_HARD)
        return FillVerdict(True, "", _string_style_score(hole, value))

    # identifier / kwarg_name / attribute_name
    if not (value.isascii() and value.isidentifier()):
        return FillVerdict(False, REJECT_GRAMMAR)
    if keyword.iskeyword(value) or keyword.issoftkeyword(value):
        return FillVerdict(False, REJECT_KEYWORD)
    if value in FORBIDDEN_NAMES or value in hole.forbidden:
        return FillVerdict(False, REJECT_FORBIDDEN)
    if hole.role in (ROLE_NEW_BINDING, ROLE_BINDING_AND_REFERENCES) \
            and value in hole.style.get("in_scope", ()):
        return FillVerdict(False, REJECT_COLLISION)

    # ---- w37 layer 2: the two measured 1B failure modes -------------------
    copy = _copy_risk(hole, value)
    generic = _generic_risk(hole, value)
    if hole.novelty == NOVELTY_REQUIRED and copy >= 1.0:
        # An exact reuse of a nearby token in a hole whose PURPOSE is to invent
        # a name. Hard only under `required` — under any other policy reusing an
        # existing identifier is frequently the correct answer.
        return FillVerdict(False, REJECT_COPY, copy_risk=copy,
                           generic_risk=generic)
    if generic >= 1.0:
        # Generic AND semantically unsupported by the role: a binding the
        # reader meets again may not be called `tmp`. A plain reference may.
        return FillVerdict(False, REJECT_DEGENERATE, copy_risk=copy,
                           generic_risk=generic)
    return FillVerdict(True, "", _identifier_style_score(hole, value),
                       copy_risk=copy, generic_risk=generic)


def _copy_risk(hole, value):
    """0..1 — how much this fill looks like a nearby token, not a new name.

    1.0 is an EXACT match against a name in scope. Below that the score is
    lexical proximity (shared segments), which is a ranking feature only: a
    name that shares a segment with its neighbours is usually GOOD style, and
    penalising it hard would select for names that do not fit the file.
    """
    peers = set(hole.style.get("in_scope", ())) | set(
        hole.style.get("peer_names", ()))
    if not peers:
        return 0.0
    if value in peers:
        return 1.0
    segs = set(_segments(value))
    if not segs:
        return 0.0
    best = 0.0
    for p in peers:
        psegs = set(_segments(p))
        if not psegs:
            continue
        best = max(best, len(segs & psegs) / float(len(segs | psegs)))
    return min(best, 0.99)


def _segments(name):
    """snake_case and camelCase segments, lowercased."""
    parts = [p for p in name.split("_") if p]
    out = []
    for p in parts:
        out.extend(s.lower() for s in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+|[A-Z]", p) or [p])
    return [s for s in out if s]


def _generic_risk(hole, value):
    """0..1. 1.0 means REJECT: generic AND unsupported by the role.

    Role-aware exactly as w37 specifies. A `reference` hole may legitimately
    resolve to `i` or `data`; a hole that introduces a binding other code will
    read may not, because the name is the only documentation that binding gets.
    """
    bare = value.lstrip("_").lower()
    if bare not in GENERIC_NAMES:
        return 0.0
    if hole.role in (ROLE_NEW_BINDING, ROLE_BINDING_AND_REFERENCES):
        return 1.0
    return 0.5


_FORMAT_RE = re.compile(r"%[-#0 +]*\d*(?:\.\d+)*[sdrfxeg%]|\{[^{}]*\}")


def _format_signature(text):
    """Ordered format placeholders. A CONTRACT, not a style preference."""
    return tuple(m.group(0) for m in _FORMAT_RE.finditer(text))


def _identifier_style_score(hole, value):
    """0..1 — SOFT. Never rejects a semantically valid fill (design ss.3E)."""
    score, weight = 0.0, 0.0
    prof = hole.style
    peers = prof.get("peer_names") or ()
    if peers:
        weight += 1.0
        snake = sum(1 for p in peers if p == p.lower())
        if (value == value.lower()) == (snake * 2 >= len(peers)):
            score += 1.0
        weight += 1.0
        priv = sum(1 for p in peers if p.startswith("_"))
        if value.startswith("_") == (priv * 2 >= len(peers)):
            score += 1.0
        lens = sorted(len(p) for p in peers)
        lo, hi = lens[0], lens[-1]
        weight += 1.0
        if lo - 4 <= len(value) <= hi + 8:
            score += 1.0
    return (score / weight) if weight else 0.5


def _string_style_score(hole, value):
    """0..1 — SOFT, mined from the file's OWN message corpus (w34 design)."""
    score, weight = 0.0, 0.0
    peers = hole.style.get("peer_messages") or ()
    if peers:
        weight += 1.0
        cap = sum(1 for p in peers if p[:1].isupper())
        if (value[:1].isupper()) == (cap * 2 >= len(peers)):
            score += 1.0
        weight += 1.0
        dot = sum(1 for p in peers if p.endswith("."))
        if value.endswith(".") == (dot * 2 >= len(peers)):
            score += 1.0
        lens = sorted(len(p) for p in peers)
        weight += 1.0
        if lens[0] * 0.4 <= len(value) <= lens[-1] * 2.5 + 20:
            score += 1.0
    return (score / weight) if weight else 0.5


# ================================================== 4. STYLE / SCOPE MINING
def _placeholder_safe(span_lines, file_lines):
    """False when the source already contains the sentinel. Fail-closed."""
    if PLACEHOLDER_PREFIX in "\n".join(span_lines):
        return False
    if file_lines and PLACEHOLDER_PREFIX in "\n".join(file_lines):
        return False
    return True


def _span_tree(span_lines):
    tree, _ = es._parse_span(span_lines)
    if tree is None:
        return None, None
    return tree, es._first_function(tree)


def scope_names(fn, file_lines=None):
    """Every name a new binding in ``fn`` could collide with. Sorted, stable."""
    names = set(es._bound_names(fn)) if fn is not None else set()
    for node in ast.walk(fn) if fn is not None else ():
        if isinstance(node, ast.Name):
            names.add(node.id)
    if file_lines:
        names |= set(es._module_bindings(file_lines) or ())
    return frozenset(names)


def peer_identifiers(fn):
    """Local binding names, nearest-first. The identifier style corpus."""
    if fn is None:
        return ()
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.append(node.id)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
    return tuple(sorted(set(out)))


def peer_messages(span_lines, file_lines=None):
    """The file's authored-message corpus (w34 ss.1): span first, file second."""
    out, seen = [], set()
    for src in ("\n".join(span_lines),
                "\n".join(file_lines) if file_lines else ""):
        if not src:
            continue
        try:
            tree = ast.parse(_dedent(src))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if 4 <= len(v) <= MAX_STRING_CHARS and " " in v \
                        and v not in seen:
                    seen.add(v)
                    out.append(v)
    return tuple(out[:40])


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


def _quote_style(span_lines):
    src = "\n".join(span_lines)
    return '"' if src.count('"') >= src.count("'") else "'"


def build_style(span_lines, file_lines, fn):
    """The `StyleProfile`, mined ONCE per span and shared by every hole."""
    return {
        "quote": _quote_style(span_lines),
        "peer_names": peer_identifiers(fn),
        "peer_messages": peer_messages(span_lines, file_lines),
        "in_scope": scope_names(fn, file_lines),
    }


# ============================================================ 5. GENERATORS
#
# Every generator is deterministic: it walks sorted AST facts and emits in a
# fixed order. None of them consults a model, a clock, a hash of anything
# mutable, or a set iteration order.


def _unique_edit(span_src, old, new, role):
    """A `SpanEdit` whose anchor is provably unique, or ``None``.

    S1 discipline 2, applied at the TEMPLATE level so an ambiguous anchor is
    rejected before a model is ever paid to fill it.
    """
    if span_src.count(old) != 1:
        return None
    return es.SpanEdit(old, new, role=role)


def _line_prefix(span_lines, node):
    """The text preceding ``node`` on its own line, or ``""``. Never raises."""
    idx = getattr(node, "lineno", 0) - 1
    if not (0 <= idx < len(span_lines)):
        return ""
    return span_lines[idx][:max(0, getattr(node, "col_offset", 0))]


def _segment(span_lines, node):
    """Exact source text of ``node`` within the span, or ``None``."""
    try:
        return ast.get_source_segment("\n".join(span_lines), node)
    except Exception:                                     # pragma: no cover
        return None


def _finish(span_lines, start_idx, edits, holes, pattern, provenance="",
            base_pattern="", substitution_only=False):
    """Build the skeleton and every gate with it, or ``None``.

    THE gate that makes a template trustworthy: the placeholder-bearing edits
    must already produce a real `EditSet`. A template that only becomes valid
    once a good fill arrives is a template that blames the model for its own
    defect.
    """
    ceiling = (MAX_SUBSTITUTION_HOLES if substitution_only
               else MAX_HOLES_PER_TEMPLATE)
    if not edits or not holes or len(holes) > ceiling:
        return None
    skeleton = es._mk_set(span_lines, start_idx, edits, pattern,
                          [pattern] + [h.hole_id for h in holes])
    if skeleton is None:
        return None
    return HoleTemplate(pattern, edits, holes, skeleton, provenance,
                        base_pattern)


def h_literal_value(span_lines, start_idx, file_lines=None, style=None,
                    base=None):
    """A string literal in the span becomes a `string_value` hole.

    The `literal_change` shape, exactly: v3 rows 8 and 9 change one authored
    prose literal and nothing else, and both are NOVEL-TOKEN by construction
    (the new text did not exist until a human wrote it). Enumeration reaches
    the STRUCTURE trivially and can never reach the TEXT — which is the precise
    division of labour this protocol exists to make.
    """
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    lits = standalone_strings(tree)
    out = []
    for i, node in enumerate(lits[:MAX_LITERAL_HOLES]):
        seg = _segment(span_lines, node)
        if seg is None or seg.startswith(("f", "b", "r", "F", "B", "R")):
            continue                      # prefixed literals get their own kind
        hstyle = dict(style)
        hstyle["format_args"] = list(_format_signature(node.value))
        hstyle["allow_newline"] = "\n" in node.value
        hole = Hole("literal_%d" % i, KIND_STRING, ROLE_MESSAGE, 0,
                    style=hstyle,
                    # Best-effort context only. `_parse_span` may have dedented,
                    # so the offsets are advisory; an out-of-range one yields ""
                    # rather than an IndexError that would kill the tranche.
                    context_before=_line_prefix(span_lines, node),
                    context_after="")
        ph = _serialize_string(placeholder(0), hole)
        ed = _unique_edit(src, seg, ph, "literal_hole")
        if ed is None:
            continue
        t = _finish(span_lines, start_idx, [ed], [hole], "hole_literal_value",
                    provenance="literal@%d" % node.lineno)
        if t is not None:
            out.append(t)
    return out


def standalone_strings(tree):
    """String Constants that are their OWN literal, sorted by position.

    EXCLUDES every Constant inside an f-string. MEASURED BUG, found by the
    oracle probe on v3 row 9 (`_compose_test`): an f-string's text parts are
    `ast.Constant` nodes whose source segment is the BARE fragment, with no
    quotes of their own. Treating one as a literal replaces
    ``...text...`` inside ``f"...text..."`` with ``"__hole0__"``, producing
    ``f""__hole0__"{x}"`` — a syntax error that the parse gate then rejected,
    silently costing the WHOLE template rather than the one bad hole.

    Filling inside an f-string is a legitimate hole kind; it is simply a
    DIFFERENT one, with its own grammar and its own serialization policy, and
    quietly approximating it with this one is how a template starts producing
    plausible but invalid output.
    """
    inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for child in ast.walk(node):
                if child is not node:
                    inner.add(id(child))
    out = [n for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and n.value.strip() and id(n) not in inner]
    return sorted(out, key=lambda n: (n.lineno, n.col_offset))


def _quoted_part(seg, value):
    """True when ``seg`` is a complete plain string literal spelling ``value``.

    Parse-based (``ast.literal_eval``), never a quote-counting heuristic: an
    implicit concatenation spanning several source lines is still one literal,
    and a heuristic that got that wrong would anchor an edit at the wrong run.
    """
    if not seg or seg[:1] not in "\"'":
        return False
    try:
        return ast.literal_eval(seg) == value
    except (ValueError, SyntaxError):
        return False


def _overlaps(a, b):
    """True when two (lineno, col, end_lineno, end_col) ranges intersect."""
    return not ((a[2], a[3]) <= (b[0], b[1]) or (b[2], b[3]) <= (a[0], a[1]))


def h_literal_multi(span_lines, start_idx, file_lines=None, style=None,
                    base=None):
    """EVERY string literal in the span becomes a hole, in ONE template.

    MEASURED, and the reason this exists: v3 row 8 (`_env_guidance`) rewrites
    TWO authored literals in one commit and row 9 (`_compose_test`) rewrites
    seven. `h_literal_value` emits one hole per template, so it can never cover
    a gold that changed two — the single-hole templates unify only if the OTHER
    literal happens to be untouched.

    Bounded by `MAX_HOLES_PER_TEMPLATE`, which is a real ceiling and not a
    formality: past three holes the model is writing the patch, and a template
    that needs seven invented strings is honestly reported as out of reach (see
    the `over the N-hole ceiling` row of ``probes/hole_reachability.py``) rather
    than reached by widening the definition of a hole.
    """
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    lits = standalone_strings(tree)
    if len(lits) < 2:
        return []                         # h_literal_value already covers one
    edits, holes, index, taken = [], [], 0, []
    for node in lits:
        if index >= MAX_SUBSTITUTION_HOLES:
            break
        seg = _segment(span_lines, node)
        if seg is None or seg.startswith(("f", "b", "r", "F", "B", "R")):
            continue
        # OVERLAP GUARD. A literal nested inside one already taken (an f-string
        # part, a docstring containing a quoted fragment) produces two edits
        # whose byte ranges intersect, and `apply_span_edits` correctly refuses
        # a write-write overlap — which silently cost the WHOLE template rather
        # than the one bad hole. Skip the nested one; keep the outer.
        rng = (node.lineno, node.col_offset,
               getattr(node, "end_lineno", node.lineno),
               getattr(node, "end_col_offset", node.col_offset))
        if any(_overlaps(rng, t) for t in taken):
            continue
        taken.append(rng)
        hstyle = dict(style)
        hstyle["format_args"] = list(_format_signature(node.value))
        hstyle["allow_newline"] = "\n" in node.value
        hole = Hole("multilit_%d" % index, KIND_STRING, ROLE_MESSAGE, index,
                    style=hstyle)
        ed = _unique_edit(src, seg, _serialize_string(placeholder(index), hole),
                          "literal_hole")
        if ed is None:
            continue
        edits.append(ed)
        holes.append(hole)
        index += 1
    if len(holes) < 2:
        return []
    t = _finish(span_lines, start_idx, edits, holes, "hole_literal_multi",
                provenance="literals x%d" % len(holes),
                substitution_only=True)
    return [t] if t is not None else []


def h_fstring_text(span_lines, start_idx, file_lines=None, style=None,
                   base=None):
    """Every TEXT RUN inside the span's f-strings becomes a hole, in ONE template.

    MEASURED: v3 row 9 (`_compose_test`) rewrites seven authored prose runs, and
    five of them live inside f-strings whose interpolations are untouched. The
    gold's AST is the parent's AST with those leaves swapped — a pure
    substitution — so it is exactly this protocol's shape, and it was invisible
    until `standalone_strings` separated the two literal kinds.

    The model fills text ONLY. Interpolations, format specs, conversions and
    quoting are structure it never sees a way to touch, and `validate_fill`
    rejects any fill containing a brace, a quote or a backslash rather than
    escaping it into something that would render differently.
    """
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    parts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for child in node.values:
            if isinstance(child, ast.Constant) and isinstance(child.value, str) \
                    and child.value.strip():
                parts.append(child)
    if not parts:
        return []
    parts.sort(key=lambda n: (n.lineno, n.col_offset,
                              getattr(n, "end_col_offset", 0)))
    edits, holes, index, seen = [], [], 0, set()
    for node in parts:
        if index >= MAX_SUBSTITUTION_HOLES:
            break
        seg = _segment(span_lines, node)
        # A part whose SOURCE spelling we cannot recover exactly (implicit
        # concatenation, escapes) is skipped, never guessed: a wrong anchor
        # here would rewrite the wrong run of a prompt.
        if not seg or seg != node.value or seg in seen:
            continue
        ph = placeholder(index)
        ed = _unique_edit(src, seg, ph, "fstring_hole")
        if ed is None:
            continue
        seen.add(seg)
        hstyle = dict(style)
        hole = Hole("ftext_%d" % index, KIND_FSTRING_TEXT, ROLE_MESSAGE, index,
                    style=hstyle)
        edits.append(ed)
        holes.append(hole)
        index += 1
    if not holes:
        return []
    t = _finish(span_lines, start_idx, edits, holes, "hole_fstring_text",
                provenance="fstring runs x%d" % len(holes),
                substitution_only=True)
    return [t] if t is not None else []


def h_substitution_all(span_lines, start_idx, file_lines=None, style=None,
                       base=None):
    """One template holing EVERY authored text leaf — plain literals AND f-string
    runs together.

    MEASURED, and the reason the two single-kind generators are not enough: v3
    row 9's gold rewrites two plain literals and five f-string runs in the same
    commit. Either single-kind template unifies with only part of it, so both
    report a miss about a gold that is, in full, a pure leaf substitution.

    Structure is untouched by construction (every edit replaces one leaf), which
    is what entitles this to `MAX_SUBSTITUTION_HOLES` rather than the structural
    ceiling.
    """
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    plain = standalone_strings(tree)
    runs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            runs.extend(c for c in node.values
                        if isinstance(c, ast.Constant)
                        and isinstance(c.value, str) and c.value.strip())
    if not plain or not runs:
        return []                          # the single-kind generators cover it
    every = sorted(plain + runs,
                   key=lambda n: (n.lineno, n.col_offset,
                                  getattr(n, "end_col_offset", 0)))
    is_run = {id(n) for n in runs}
    edits, holes, index, taken, seen = [], [], 0, [], set()
    for node in every:
        if index >= MAX_SUBSTITUTION_HOLES:
            break
        seg = _segment(span_lines, node)
        if not seg:
            continue
        rng = (node.lineno, node.col_offset,
               getattr(node, "end_lineno", node.lineno),
               getattr(node, "end_col_offset", node.col_offset))
        if any(_overlaps(rng, t) for t in taken):
            continue
        if id(node) in is_run:
            # A JoinedStr's parts come in TWO source shapes, and conflating
            # them is the same class of bug `standalone_strings` fixed:
            #   bare run   f"...TEXT..."          seg IS the text, no quotes
            #   quoted part f"{x}" "TEXT"         implicit concatenation of a
            #               plain literal INTO the f-string — ast folds it into
            #               the same JoinedStr, but its source segment is a
            #               complete quoted literal and it must be filled as one.
            # MEASURED on v3 row 9, whose two largest authored blocks are the
            # quoted kind; treating them as bare runs skipped them silently and
            # the template then unified with nothing.
            quoted = _quoted_part(seg, node.value)
            if seg == node.value and seg not in seen:
                kind, new = KIND_FSTRING_TEXT, placeholder(index)
            elif quoted:
                kind, new = KIND_STRING, None
            else:
                continue
        else:
            if seg.startswith(("f", "b", "r", "F", "B", "R")):
                continue
            kind = KIND_STRING
            new = None                     # needs the hole to know the quote
        hstyle = dict(style)
        hstyle["format_args"] = list(_format_signature(node.value))
        hstyle["allow_newline"] = "\n" in node.value
        if kind == KIND_STRING and seg[:1] in "\"'":
            hstyle["quote"] = seg[0]      # preserve the part's OWN quote style
        hole = Hole("sub_%d" % index, kind, ROLE_MESSAGE, index, style=hstyle)
        if new is None:
            new = _serialize_string(placeholder(index), hole)
        ed = _unique_edit(src, seg, new, "substitution_hole")
        if ed is None:
            continue
        taken.append(rng)
        seen.add(seg)
        edits.append(ed)
        holes.append(hole)
        index += 1
    if len(holes) < 2:
        return []
    t = _finish(span_lines, start_idx, edits, holes, "hole_substitution_all",
                provenance="text leaves x%d" % len(holes),
                substitution_only=True)
    return [t] if t is not None else []


def h_attribute_name(span_lines, start_idx, file_lines=None, style=None,
                     base=None):
    """An attribute access becomes an `attribute_name` hole (attribute_swap)."""
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    attrs = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Attribute)),
                   key=lambda n: (n.lineno, n.col_offset))
    out = []
    for i, node in enumerate(attrs):
        seg = _segment(span_lines, node)
        base_seg = _segment(span_lines, node.value)
        if not seg or not base_seg or not seg.endswith("." + node.attr):
            continue
        hole = Hole("attr_%d" % i, KIND_ATTRIBUTE, ROLE_ATTRIBUTE, 0,
                    forbidden=(node.attr,), style=style,
                    context_before=base_seg + ".")
        ed = _unique_edit(src, seg, "%s.%s" % (base_seg, placeholder(0)),
                          "attribute_hole")
        if ed is None:
            continue
        t = _finish(span_lines, start_idx, [ed], [hole], "hole_attribute_name",
                    provenance="attr@%d:%s" % (node.lineno, node.attr))
        if t is not None:
            out.append(t)
    return out


def h_kwarg_name(span_lines, start_idx, file_lines=None, style=None,
                 base=None):
    """A call's keyword-argument NAME becomes a `kwarg_name` hole."""
    tree, fn = _span_tree(span_lines)
    if tree is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    out, i = [], 0
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)),
                   key=lambda n: (n.lineno, n.col_offset))
    for call in calls:
        siblings = tuple(k.arg for k in call.keywords if k.arg)
        for kw in call.keywords:
            if kw.arg is None:
                continue
            val = _segment(span_lines, kw.value)
            if val is None:
                continue
            old = "%s=%s" % (kw.arg, val)
            hole = Hole("kwarg_%d" % i, KIND_KWARG, ROLE_KWARG, 0,
                        forbidden=siblings, style=style,
                        context_before="", context_after="=" + val)
            i += 1
            ed = _unique_edit(src, old, "%s=%s" % (placeholder(0), val),
                              "kwarg_hole")
            if ed is None:
                continue
            t = _finish(span_lines, start_idx, [ed], [hole], "hole_kwarg_name",
                        provenance="kwarg@%d:%s" % (kw.value.lineno, kw.arg))
            if t is not None:
                out.append(t)
    return out


def h_hoist_binding(span_lines, start_idx, file_lines=None, style=None,
                    base=None):
    """Hoist a repeated sub-expression to an INVENTED local.

    This is `edit_sets.p2_hoist_shared_subexpr`'s shape with the one difference
    that matters: p2 names the binding after the dict key, so it can only
    propose a name the parent file already contains, and it BAILS entirely when
    that name is taken. The measured golds (``_cc``, ``launched``,
    ``_claimed_author``, ``_created_by``) all invent a name instead — which is
    why every one of them sits in `PUNTED_PATTERNS` as NOVEL-TOKEN.

    Here the name is a `binding_and_references` hole. The generator therefore
    fires in the cases p2 correctly refuses, and it never competes with p2 on
    the cases p2 already wins (p2's candidate is emitted first and this one is
    a distinct pattern, so the existing menu keeps its exact order).
    """
    tree, fn = _span_tree(span_lines)
    if tree is None or fn is None:
        return []
    src = "\n".join(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    parents = es._parent_map(fn)
    groups = {}
    for node in ast.walk(fn):
        b = es._get_call_base(node)
        if b is not None:
            groups.setdefault(b, []).append(node)
    out = []
    for (obj, key), seeds in sorted(groups.items()):
        if len(seeds) < 2:
            continue
        seeds = sorted(seeds, key=lambda n: (n.lineno, n.col_offset))
        texts = []
        for s in seeds:
            t = es._expr_text(span_lines,
                              es._climb(parents, s, through_comp=False))
            if not t:
                texts = []
                break
            texts.append(t)
        if not texts or len(set(texts)) != len(texts):
            continue
        if any(src.count(t) != 1 for t in texts):
            continue
        anchor_i = es._hoist_anchor(fn, seeds[0].lineno)
        if anchor_i is None:
            continue
        aline = span_lines[anchor_i]
        if src.count(aline + "\n") != 1:
            continue
        ind = es._indent_of(aline)
        canonical = texts[0]
        ph = placeholder(0)
        edits = [es.SpanEdit(t, ph, role="retarget_consumer")
                 for t in sorted(texts, key=lambda t: (-len(t), t))]
        edits.append(es.SpanEdit(
            aline + "\n", "%s\n%s%s = %s\n" % (aline, ind, ph, canonical),
            role="hoist"))
        hstyle = dict(style)
        hole = Hole("binding_%s" % key, KIND_IDENTIFIER,
                    ROLE_BINDING_AND_REFERENCES, 0,
                    occurrences=len(texts) + 1,
                    forbidden=(), style=hstyle,
                    context_before="%s = %s" % ("<name>", canonical))
        t = _finish(span_lines, start_idx, edits, [hole],
                    "hole_hoist_binding",
                    provenance="hoist:%s.%s" % (obj, key))
        if t is not None:
            out.append(t)
    return out


def h_candidate_literal(span_lines, start_idx, file_lines=None, style=None,
                        base=None):
    """Hole-variants of an ALREADY-ENUMERATED structural candidate.

    The measured shape of most novel-token misses is not "a wrong structure" but
    "the right structure carrying prose a human authored" — v3 rows 3, 7, 10 and
    the two `authored_message_string` census punts all insert a statement whose
    only unreachable token is its message. `c2_statement_insert` and
    `c3_exception_handler` already produce the statement; what they cannot
    produce is the sentence.

    So this generator takes the base menu as INPUT and re-emits each candidate
    with its inserted string literals turned into typed holes. It adds no
    structural reach of its own, which is exactly why it is safe: the structure
    was already gated by the family that produced it.
    """
    if not base:
        return []
    tree, fn = _span_tree(span_lines)
    style = style or build_style(span_lines, file_lines, fn)
    out = []
    for cand in list(base)[:MAX_BASE_CANDIDATES]:
        edits = getattr(cand, "edits", None)
        pattern = getattr(cand, "pattern", "")
        if not edits:
            continue
        # First-seen order across the set's edits, deduped. Deterministic, and
        # deliberately NOT sorted: the order the generator emitted its edits in
        # is the order a reader sees the statement, so hole 0 is the first
        # message the inserted code would print.
        seen = []
        for ed in edits:
            for lit in _literals_in(ed.new):
                if lit not in seen:
                    seen.append(lit)
        for i, lit in enumerate(seen[:MAX_LITERAL_HOLES]):
            hstyle = dict(style)
            hstyle["format_args"] = list(_format_signature(lit))
            hstyle["allow_newline"] = "\n" in lit
            hole = Hole("msg_%d" % i, KIND_STRING, ROLE_MESSAGE, 0,
                        style=hstyle)
            ph = _serialize_string(placeholder(0), hole)
            new_edits, hit = [], False
            for ed in edits:
                new_text, n = _replace_literal(ed.new, lit, ph)
                hit = hit or bool(n)
                new_edits.append(es.SpanEdit(ed.old, new_text, ed.role))
            if not hit:
                continue
            t = _finish(span_lines, start_idx, new_edits, [hole],
                        "hole_%s_message" % pattern,
                        provenance="from:%s" % pattern, base_pattern=pattern)
            if t is not None:
                out.append(t)
    return out


def _literals_in(text):
    """String-literal VALUES inside a code fragment. Parse-based, never regex."""
    try:
        tree = ast.parse(_dedent(text))
    except SyntaxError:
        return ()
    vals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.strip():
            vals.append(node.value)
    return tuple(vals)


def _replace_literal(text, value, replacement):
    """Replace every SOURCE spelling of ``value`` in ``text``. (text, count)."""
    n = 0
    for quote in ('"', "'"):
        spelled = "%s%s%s" % (quote, value.replace("\\", "\\\\")
                              .replace(quote, "\\" + quote)
                              .replace("\n", "\\n"), quote)
        if spelled in text:
            n += text.count(spelled)
            text = text.replace(spelled, replacement)
    return text, n


#: THE HOLE PUNTS — declared with the measured token that blocks each, in the
#: same discipline as `edit_sets.PUNTED_PATTERNS` and `CENSUS_PUNTS`. Printed by
#: ``probes/hole_reachability.py`` so a miss is never silently missing.
#:
#: MEASURED RESULT on the v3 real board: 17 of 21 rows carry a novel token; 2 of
#: those 17 differ from their parent ONLY in hole-shaped leaves; the templates
#: reach 1 of the 2 with a perfect fill. The headline finding is NOT that the
#: fills are weak — it is that on this board the binding constraint is
#: STRUCTURE, not the token. Fifteen of seventeen golds add statements, guards
#: or handlers that no template generator builds, and a perfect model filling a
#: perfect hole would still miss every one of them.
HOLE_PUNTS = {
    "structure_not_token": (
        "STRUCTURAL. 15 of the 17 novel-token v3 rows add STATEMENTS the "
        "enumerator does not build — a 5-line sanitize block (row 14), a "
        "15-line safety-field re-validation (row 18), a loop guard with its own "
        "comment (row 16). The novel token is real but it is not what blocks "
        "them: erase the token problem entirely and they are still out of "
        "reach. Typed holes are the wrong lever for these; a structural "
        "generator is the next build order."),
    "folded_fstring_part": (
        "SOURCE-RECOVERY. v3 row 9 (`_compose_test`) is a pure leaf "
        "substitution — its gold AST is the parent's with seven authored text "
        "leaves swapped — but five of them live in an f-string implicitly "
        "concatenated with plain literals. CPython folds those into ONE "
        "`Constant` whose `get_source_segment` is neither the bare text nor a "
        "complete literal, so no anchor can be recovered exactly. "
        "`h_substitution_all` deliberately SKIPS such a part rather than "
        "guessing: a wrong anchor here rewrites the wrong run of a prompt. "
        "Reaching it needs f-string source tokenization, which is a real build "
        "and not a widened heuristic."),
    "masked_context_second_view": (
        "NOT BUILT, DECLARED. w37 recommends a second prompt VIEW with nearby "
        "identifier lexemes masked (types and roles retained): a fill that "
        "changes completely between the full-context and masked views has "
        "copy-attractor risk that no single-view sampling can see. Shipped "
        "instead: an exclusion-framed prompt under `novelty=required` plus a "
        "hard `REJECT_COPY`, which catches EXACT reuse but not semantic "
        "copying. The second view doubles the per-hole spend for a signal "
        "nothing on this board has yet been measured to need, and building it "
        "without that measurement would be buying a mechanism on a rows-0 "
        "board."),
    "over_the_structural_ceiling": (
        "BY DESIGN. 6 rows invent more tokens than `MAX_HOLES_PER_TEMPLATE` "
        "allows a structural template to carry. Raising the ceiling would not "
        "reach them either (their structure is also unreachable), and past "
        "three holes the model is writing the patch — which is the one thing "
        "this protocol exists to prevent. Recorded as a ceiling, not a defeat."),
}

#: Fixed order. Appended-last discipline: a new generator goes at the END so
#: every template menu measured before it keeps its exact order.
TEMPLATE_GENERATORS = (
    ("hole_literal_value", h_literal_value),
    ("hole_literal_multi", h_literal_multi),
    ("hole_fstring_text", h_fstring_text),
    ("hole_substitution_all", h_substitution_all),
    ("hole_attribute_name", h_attribute_name),
    ("hole_hoist_binding", h_hoist_binding),
    ("hole_kwarg_name", h_kwarg_name),
    ("hole_candidate_message", h_candidate_literal),
)


# ============================================================== 6. ENTRY POINT
def enumerate_hole_templates(span_lines, start_idx, file_lines=None,
                             base_candidates=(), cap=None):
    """``(templates, metadata)``. Deterministic, bounded, and self-reporting.

    Round-robin across generators for the same reason `_run_tranche` does it:
    concatenation order is not a relevance order, and a generator with a big
    output would otherwise eat the head of the cap and push every later
    generator's BEST template below its WORST.
    """
    cap = MAX_HOLE_TEMPLATES if cap is None else cap
    meta = {
        "emitted": 0, "explored": 0, "truncated": False,
        "generators_failed": [], "placeholder_unsafe": False,
        "negative_conclusion_allowed": False,
    }
    if not _placeholder_safe(span_lines, file_lines):
        # Fail-closed and REPORTED: a source that already contains the sentinel
        # cannot be templated safely, and silently returning [] would launder
        # that into "this function has no typed holes".
        meta["placeholder_unsafe"] = True
        return [], meta
    tree, fn = _span_tree(span_lines)
    style = build_style(span_lines, file_lines, fn)
    produced, failed = [], []
    for name, gen in TEMPLATE_GENERATORS:
        try:
            produced.append(list(gen(span_lines, start_idx, file_lines,
                                     style, base_candidates)))
        except Exception as exc:          # law 1: rather learn nothing
            failed.append("%s:%s" % (name, type(exc).__name__))
            produced.append([])
    out, seen, explored, truncated = [], set(), 0, False
    rank = 0
    while not truncated:
        progressed = False
        for cands in produced:
            if rank >= len(cands):
                continue
            progressed = True
            explored += 1
            if explored > MAX_HOLE_EXPLORED:
                truncated = True
                break
            t = cands[rank]
            if t.template_id in seen:
                continue
            seen.add(t.template_id)
            out.append(t)
        if not progressed:
            break
        rank += 1
    if len(out) > cap:
        truncated = True
    out = out[:cap]
    meta.update({
        "emitted": len(out), "explored": explored, "truncated": truncated,
        "generators_failed": failed,
        "negative_conclusion_allowed": (not truncated) and not failed,
        "holes": sum(t.n_holes for t in out),
        "patterns": sorted({t.pattern for t in out}),
    })
    return out, meta


# ========================================================= 7. THE ORACLE FILL
#
# Requirement 5, and the only honest way to answer "is the template wrong or is
# the model wrong". `oracle_fill` injects the GOLD token into the hole with NO
# model in the loop. What it measures is TEMPLATE COVERAGE alone. It is an
# offline instrument: it takes gold as input and therefore may never run inside
# a fix, a selection policy, or anything that scores a model.


class _Unifier:
    """Structural unification of a placeholder-bearing AST against gold."""

    def __init__(self):
        self.fills = {}

    def unify(self, cand, gold):
        if type(cand) is not type(gold):
            # A placeholder may stand in for a leaf of a DIFFERENT node type
            # (a Name hole against a gold Attribute, say). `_hole` returns None
            # when `cand` is not a hole at all, and a type mismatch that is not
            # a hole is a plain rejection — never None, which a caller would
            # read as falsy anyway but for the wrong reason.
            return bool(self._hole(cand, gold))
        hit = self._hole(cand, gold)
        if hit is not None:
            return hit
        if isinstance(cand, ast.AST):
            for field in cand._fields:
                if not self.unify(getattr(cand, field, None),
                                  getattr(gold, field, None)):
                    return False
            return True
        if isinstance(cand, list):
            if len(cand) != len(gold):
                return False
            return all(self.unify(a, b) for a, b in zip(cand, gold))
        return cand == gold

    def _hole(self, cand, gold):
        """True/False if ``cand`` is a hole, else ``None`` (not a hole)."""
        tok = _hole_token_of(cand)
        if tok is None:
            return None
        value = _leaf_value_of(gold, cand)
        if value is None:
            return False
        prev = self.fills.get(tok)
        if prev is not None and prev != value:
            return False                  # one hole, one value. Always.
        self.fills[tok] = value
        return True


def _hole_token_of(node):
    """The placeholder this node carries, or ``None``."""
    if isinstance(node, ast.Name) and _PLACEHOLDER_RE.fullmatch(node.id):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str) \
            and _PLACEHOLDER_RE.fullmatch(node.value):
        return node.value
    if isinstance(node, ast.Attribute) and _PLACEHOLDER_RE.fullmatch(node.attr):
        return node.attr
    if isinstance(node, str) and _PLACEHOLDER_RE.fullmatch(node):
        return node
    return None


def _leaf_value_of(gold, cand):
    """The gold value a hole of ``cand``'s KIND may absorb, or ``None``.

    Kind is enforced: a string hole may not absorb an identifier and an
    identifier hole may not absorb a literal. Without this, a template would
    "reach" gold through a type-confused substitution that could never be
    produced by the real fill path — an oracle that flatters the template.
    """
    if isinstance(cand, ast.Name):
        return gold.id if isinstance(gold, ast.Name) else None
    if isinstance(cand, ast.Constant):
        return (gold.value if isinstance(gold, ast.Constant)
                and isinstance(gold.value, str) else None)
    if isinstance(cand, ast.Attribute):
        return gold.attr if isinstance(gold, ast.Attribute) else None
    if isinstance(cand, str):
        return gold if isinstance(gold, str) else None
    return None


def oracle_fill(template, span_lines, start_idx, gold_src):
    """``fills`` (hole index -> gold value) that make the template EQUAL gold.

    ``None`` when no assignment does — which is the honest answer "this template
    does not cover this gold", distinct from "the model could not fill it".

    OFFLINE ONLY. Never call this from a fix path: it reads gold.
    """
    skeleton_src = _instantiated_span(template, span_lines)
    if skeleton_src is None:
        return None
    try:
        cand_tree = ast.parse(_dedent(skeleton_src))
        gold_tree = ast.parse(_dedent(gold_src))
    except SyntaxError:
        return None
    u = _Unifier()
    if not u.unify(cand_tree, gold_tree):
        return None
    out = {}
    for hole in template.holes:
        if hole.token not in u.fills:
            # A hole the gold never constrained: the template is over-general
            # here, and reporting reachability for it would be a free pass.
            return None
        out[hole.index] = u.fills[hole.token]
    return out


def _instantiated_span(template, span_lines):
    """The span with the SKELETON applied (placeholders live). Source, or None."""
    applied = es.apply_to_span(span_lines, template.skeleton)
    if applied is None:
        return None
    return "\n".join(applied)


def oracle_reachable(template, span_lines, start_idx, gold_src):
    """``(reachable, fills, verdicts)`` — coverage AND whether the fill is legal.

    A template that reaches gold only with a fill its OWN validator would
    reject is NOT reachable, and saying otherwise would report a ceiling the
    protocol can never touch. Both halves are returned so the probe can report
    them separately rather than collapsing them.
    """
    fills = oracle_fill(template, span_lines, start_idx, gold_src)
    if fills is None:
        return False, None, {}
    verdicts = {}
    for hole in template.holes:
        verdicts[hole.index] = validate_fill(hole, fills[hole.index], template)
    ok = all(v.ok for v in verdicts.values())
    if not ok:
        return False, fills, verdicts
    built = instantiate(template, fills, span_lines, start_idx)
    if built is None:
        return False, fills, verdicts
    applied = es.apply_to_span(span_lines, built)
    if applied is None:
        return False, fills, verdicts
    try:
        same = (ast.dump(ast.parse(_dedent("\n".join(applied))))
                == ast.dump(ast.parse(_dedent(gold_src))))
    except SyntaxError:
        return False, fills, verdicts
    return same, fills, verdicts


# ================================================== 8. NEAR-MISS CLASSIFICATION
#
# Requirement 3. The strict metric does NOT move: `exact` is byte-for-byte AST
# equality and stays that way. This adds a SEPARATE, explicitly-labelled record
# so the bench can say "the model built the right thing and spelled it
# differently" without ever letting that count as a pass.

NEAR_MISS_NONE = ""
NEAR_MISS_RENAME = "near_miss_token_only:rename"
NEAR_MISS_STRING = "near_miss_token_only:string_content"
NEAR_MISS_BOTH = "near_miss_token_only:rename+string_content"


class _Canon(ast.NodeTransformer):
    """Erase exactly the two token classes a hole can produce, nothing else."""

    def __init__(self, names, strings):
        self.names, self.strings = names, strings
        self.renamed, self.restrung = False, False
        self._map = {}

    def visit_Name(self, node):
        if self.names:
            node.id = self._canon(node.id)
        return node

    def visit_Constant(self, node):
        if self.strings and isinstance(node.value, str):
            node.value = "<STR>"
        return node

    def _canon(self, name):
        if name not in self._map:
            self._map[name] = "<N%d>" % len(self._map)
        return self._map[name]


def _canonical(src, names, strings):
    try:
        return ast.dump(_Canon(names, strings).visit(ast.parse(_dedent(src))))
    except SyntaxError:
        return None


def classify_near_miss(candidate_src, gold_src):
    """`near_miss_token_only` label for a NON-exact candidate, or ``""``.

    Returns "" for an exact match too: an exact match is not a near miss, and a
    cert that carried both would let a reader double-count it.

    The labels are deliberately narrow. `rename` means the two functions are
    AST-equal modulo a CONSISTENT bijective renaming of local names — the
    ``_cc`` vs ``_claimed_author`` case. `string_content` means they are
    AST-equal modulo string literal CONTENT — the ``'Turn dispatch failed'``
    case. Anything else is not a near miss, and this function says so rather
    than reaching for a similarity score.
    """
    exact = _canonical(candidate_src, False, False)
    gold_exact = _canonical(gold_src, False, False)
    if exact is None or gold_exact is None:
        return NEAR_MISS_NONE
    if exact == gold_exact:
        return NEAR_MISS_NONE                      # exact: not a near miss
    rename = (_canonical(candidate_src, True, False)
              == _canonical(gold_src, True, False))
    string = (_canonical(candidate_src, False, True)
              == _canonical(gold_src, False, True))
    both = (_canonical(candidate_src, True, True)
            == _canonical(gold_src, True, True))
    if rename and string:
        return NEAR_MISS_BOTH
    if rename:
        return NEAR_MISS_RENAME
    if string:
        return NEAR_MISS_STRING
    if both:
        return NEAR_MISS_BOTH
    return NEAR_MISS_NONE


# ==================================================== 9. THE FILL CALL (model)
#
# One short, no-think completion per hole. The model NEVER sees gold, never sees
# the hidden tests, and is never asked to "recover the missing name" — it is
# asked to write a name that fits. It cannot emit a patch because the protocol
# gives it nowhere to put one.


GBNF_IDENTIFIER = """\
root ::= ident
ident ::= [A-Za-z_] [A-Za-z0-9_]*
"""

GBNF_STRING = """\
root ::= text
text ::= [^\\x00-\\x1F\\x7F"\\\\]+
"""

#: kwarg and attribute names are identifiers; one grammar, three kinds.
GRAMMARS = {
    KIND_IDENTIFIER: GBNF_IDENTIFIER,
    KIND_KWARG: GBNF_IDENTIFIER,
    KIND_ATTRIBUTE: GBNF_IDENTIFIER,
    KIND_STRING: GBNF_STRING,
    #: An f-string run additionally forbids braces at the GRAMMAR level, since a
    #: brace there is an interpolation — the model writing code.
    KIND_FSTRING_TEXT: "root ::= text\ntext ::= [^\\x00-\\x1F\\x7F\"'\\\\{}]+\n",
}


def grammar_for(hole):
    """The GBNF for this hole's kind. Advisory: `validate_fill` is authority.

    The grammar cannot express keyword exclusion, scope collision, or style, and
    on current llama.cpp builds it is silently INACTIVE whenever thinking is
    enabled (`gym_config`'s grammar_pick lesson). Both facts point the same way:
    the post-validator decides, and a grammar only makes the common case cheap.
    """
    return GRAMMARS.get(hole.kind)


def build_fill_prompt(template, hole, span_lines, file_lines=None):
    """The complete model input for ONE hole. Deterministic given its inputs.

    Everything the model is allowed to know, and nothing else: the code with the
    hole marked, the slot's kind and role, and nearby style examples. It does
    not receive the gold token, the gold patch, the tests, or an instruction to
    recover anything.
    """
    marked = _instantiated_span(template, span_lines) or "\n".join(span_lines)
    marked = marked.replace(hole.token, "<<<FILL>>>")
    lines = ["Code with one blank marked <<<FILL>>>:", "", marked, ""]
    if hole.kind == KIND_STRING:
        peers = list(hole.style.get("peer_messages") or ())[:6]
        lines.append("Write the message text for <<<FILL>>>. Plain text only, "
                     "no quotes, one line.")
        if hole.style.get("format_args"):
            lines.append("It MUST contain exactly these format placeholders, "
                         "in order: %s"
                         % " ".join(hole.style["format_args"]))
        if peers:
            lines.append("Nearby messages in this file, for tone and shape:")
            lines.extend("  %s" % p for p in peers)
    else:
        peers = list(hole.style.get("peer_names") or ())[:12]
        lines.append("Write the %s for <<<FILL>>> (role: %s). "
                     "A bare Python identifier, nothing else."
                     % (hole.kind.replace("_", " "), hole.role))
        if peers:
            # HOW the peers are framed depends on the novelty policy, and it
            # matters: showing a 1B a list of nearby names and asking it to
            # invent one is a copy attractor by construction (w37's first
            # failure mode). Under `required` the same list is presented as the
            # EXCLUSION set it actually is.
            if hole.novelty == NOVELTY_REQUIRED:
                lines.append("These names are ALREADY TAKEN and must not be "
                             "reused — use them only to match the local naming "
                             "style: %s" % ", ".join(peers))
                lines.append("Write a NEW name that says what this value IS. "
                             "Do not answer tmp, foo, data, val or result.")
            else:
                lines.append("Nearby names in this scope: %s" % ", ".join(peers))
        if hole.role in (ROLE_NEW_BINDING, ROLE_BINDING_AND_REFERENCES):
            lines.append("It must NOT collide with any name already in scope.")
    lines.append("Answer with the value alone.")
    return "\n".join(lines)


def sample_fills(hole, template, span_lines, file_lines=None, complete=None,
                 samples=FILL_SAMPLES, budget=None):
    """``(values, telemetry)`` — up to ``samples`` DEDUPED, VALID fills.

    ``complete(prompt, grammar=..., max_tokens=..., seed=...) -> str`` is the
    injected model seam, so every test in this module runs offline against a
    stub and the protocol is provable without a GPU.

    A sample that fails validation is COUNTED AS A FAILED SAMPLE and discarded.
    It is never retried into existence: retrying until something valid appears
    would make pass@N dishonest and the spend unbounded.
    """
    tel = {"requested": 0, "returned": 0, "valid": 0, "rejected": {},
           "budget_exhausted": False, "call_failed": 0}
    if complete is None:
        return (), tel
    samples = samples_for(hole, samples)
    prompt = build_fill_prompt(template, hole, span_lines, file_lines)
    grammar = grammar_for(hole)
    out, tally, features = [], {}, {}
    for i in range(samples):
        if budget is not None and not budget.take():
            tel["budget_exhausted"] = True
            break
        tel["requested"] += 1
        try:
            raw = complete(
                prompt, grammar=grammar,
                # ANSWER tokens, not the server's max_tokens. The seam adds the
                # #11d reasoning headroom; handing a raw answer size straight to
                # llama.cpp is the guillotine that rule exists to stop.
                answer_tokens=(256 if hole.kind in (KIND_STRING,
                                                    KIND_FSTRING_TEXT) else 16),
                seed=_seed(template.template_id, hole.index, i))
        except Exception as exc:          # law 1: a dead provider is REPORTED
            tel["call_failed"] += 1
            tel.setdefault("errors", []).append(type(exc).__name__)
            continue
        value = _decode(raw, hole)
        if value is None:
            tel["rejected"]["REJECT_PROTOCOL"] = \
                tel["rejected"].get("REJECT_PROTOCOL", 0) + 1
            continue
        tel["returned"] += 1
        verdict = validate_fill(hole, value, template)
        if not verdict.ok:
            tel["rejected"][verdict.reason] = \
                tel["rejected"].get(verdict.reason, 0) + 1
            continue
        tally[value] = tally.get(value, 0) + 1
        features.setdefault(value, verdict.features())
        if value not in out:
            out.append(value)
    tel["valid"] = len(out)
    tel.update(agreement(tally, tel["requested"]))
    tel["features"] = features
    # Deterministic order: most-agreed first, then first-seen. Selection still
    # belongs to the pipeline's own arbitration — this only decides which fill
    # heads the candidate product when the budget truncates it.
    order = {v: i for i, v in enumerate(out)}
    out.sort(key=lambda v: (-tally.get(v, 0), order[v]))
    return tuple(out), tel


def samples_for(hole, default=FILL_SAMPLES):
    """How many samples this hole's RISK justifies (w37 ss.3).

    An invention hole carries the copy-attractor, degenerate-name and
    semantic-quality risks all at once and gets more looks; a reference hole is
    dominated by deterministic constraints and gets fewer. Never fewer than one,
    never more than double the configured default, so a caller's budget still
    means something.
    """
    if hole.role in (ROLE_NEW_BINDING, ROLE_BINDING_AND_REFERENCES):
        return max(1, min(default * 2, default + 2))
    if hole.role == ROLE_REFERENCE:
        return max(1, default - 1)
    return max(1, default)


def agreement(tally, requested):
    """Stability statistics over the samples. NOT a probability of correctness.

    w37's core rule, and the one this function refuses to blur:

        agreement increases confidence in a candidate's STABILITY;
        it does not establish CORRECTNESS.

    Two consequences are built in rather than commented:

    * INVALID SAMPLES STAY IN THE DENOMINATOR. "4 identical valid out of 5" and
      "4 identical valid out of 4" are different facts, and a rate computed over
      survivors only would report them as the same.
    * The key is `mode_rate`, never `probability`. These are temperature-0
      completions of one prompt; they are correlated by construction and calling
      them independent votes would be inventing evidence. Treat the number as a
      ranking feature to be calibrated against adjudicated outcomes, exactly as
      the 1B menu-confidence protocol does — not as a posterior.
    """
    if not requested:
        return {"mode_rate": 0.0, "valid_rate": 0.0, "margin": 0.0,
                "mode": None, "mode_count": 0}
    counts = sorted(tally.values(), reverse=True)
    top = counts[0] if counts else 0
    second = counts[1] if len(counts) > 1 else 0
    mode = None
    if tally:
        mode = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return {
        "mode": mode,
        "mode_count": top,
        "mode_rate": top / float(requested),
        "valid_rate": sum(tally.values()) / float(requested),
        "margin": (top - second) / float(requested),
        # Named so no reader can mistake it for P(correct).
        "agreement_is_stability_not_correctness": True,
    }


def _seed(template_id, hole_index, sample_index):
    """Deterministic per-sample seed: same run, same seeds, same samples."""
    h = hashlib.sha256(("%s|%d|%d" % (template_id, hole_index, sample_index))
                       .encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")


def _decode(raw, hole):
    """The model's ENVELOPE. One value, or ``None``. Never a partial parse."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("<think>"):
        # A think block from a model that ignored the no-think setting. Take
        # what follows and nothing inside it — never treat reasoning as an answer.
        end = text.find("</think>")
        if end < 0:
            return None
        text = text[end + len("</think>"):].strip()
    if not text:
        return None
    text = text.strip("`").strip()
    if hole.kind == KIND_STRING:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1]
        return text.split("\n")[0] if not hole.style.get("allow_newline") \
            else text
    return text.split()[0] if text.split() else None


class FillBudget:
    """A hard, shared, REPORTED cap on model calls for one task."""

    __slots__ = ("limit", "used")

    def __init__(self, limit=MAX_FILL_CALLS):
        self.limit = limit
        self.used = 0

    def take(self):
        if self.used >= self.limit:
            return False
        self.used += 1
        return True

    @property
    def remaining(self):
        return max(0, self.limit - self.used)

    def telemetry(self):
        return {"fill_calls_limit": self.limit, "fill_calls_used": self.used,
                "fill_calls_remaining": self.remaining}


def fill_templates(templates, span_lines, start_idx, file_lines=None,
                   complete=None, samples=FILL_SAMPLES, budget=None,
                   max_templates=None):
    """``(candidates, telemetry)`` — every template, every valid fill.

    Each returned candidate is an ORDINARY `edit_sets.EditSet`. It carries no
    special status, no bypass, and no priority: it enters spec_exec, witness and
    ballot on exactly the terms an enumerated candidate does. That is the whole
    safety argument — by the time arbitration sees a fill, arbitration cannot
    tell it was one, and does not need to.

    Multi-hole templates take the CARTESIAN product of their per-hole fills,
    bounded by `MAX_HOLES_PER_TEMPLATE` and the shared budget. With N=3 and the
    3-hole ceiling that is at most 27 candidates from one template, which the
    menu cap then trims.
    """
    budget = budget if budget is not None else FillBudget()
    tel = {"templates": 0, "candidates": 0, "per_hole": [],
           "rejected": {}, "instantiate_failed": 0}
    out = []
    limit = MAX_HOLE_TEMPLATES if max_templates is None else max_templates
    for template in list(templates)[:limit]:
        tel["templates"] += 1
        per_hole = []
        for hole in template.holes:
            values, htel = sample_fills(hole, template, span_lines, file_lines,
                                        complete=complete, samples=samples,
                                        budget=budget)
            htel["hole_id"] = hole.hole_id
            htel["kind"] = hole.kind
            tel["per_hole"].append(htel)
            for reason, n in htel["rejected"].items():
                tel["rejected"][reason] = tel["rejected"].get(reason, 0) + n
            if not values:
                per_hole = []
                break
            per_hole.append((hole, values))
        if not per_hole:
            continue
        combos, combo_trunc = _product(per_hole)
        if combo_trunc:
            tel["combos_truncated"] = tel.get("combos_truncated", 0) + 1
        for combo in combos:
            built = instantiate(template, combo, span_lines, start_idx)
            if built is None:
                tel["instantiate_failed"] += 1
                continue
            if built not in out:
                out.append(built)
    tel["candidates"] = len(out)
    tel.update(budget.telemetry())
    return out, tel


#: Candidates one template may contribute. A `MAX_SUBSTITUTION_HOLES`-wide
#: template at N=3 has 6,561 combinations, which is an exponential wearing a
#: menu's clothes. The product is taken in a fixed order (first sample of every
#: hole first, so the model's top pick for each slot is always among the kept
#: combos) and TRUNCATED here, reported via `combos_truncated`.
MAX_CANDIDATES_PER_TEMPLATE = 27


def _product(per_hole, cap=MAX_CANDIDATES_PER_TEMPLATE):
    """Deterministic, BOUNDED Cartesian product over (hole, values) pairs.

    Returns ``(combos, truncated)``. Order is odometer-style with the LAST hole
    varying fastest, so combo 0 is every hole's first sample — never a random
    survivor of a truncation.
    """
    combos, truncated = [{}], False
    for hole, values in per_hole:
        nxt = []
        for base in combos:
            for v in values:
                d = dict(base)
                d[hole.index] = v
                nxt.append(d)
        if len(nxt) > cap:
            nxt = nxt[:cap]
            truncated = True
        combos = nxt
    return combos, truncated
