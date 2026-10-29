"""Single-line mutation-candidate enumerator (the closed candidate set).

Extracted from probes/real_repo_baseline_v3.py so the fixer, the v3 probe and the
bench can all share ONE authority, then extended with three new complete families.

Contract (unchanged): enumerate_mutations(span_lines, start_idx, cls) returns a
deduped list of (line_index, old_line, new_line) tuples, where line_index is an
absolute 0-based index into the ORIGINAL file lines (start_idx + offset in span).

Families
--------
legacy (reuse the harvest injectors so a flip on a bugged line restores the original):
  cmp_flip     comparison operators, FULL substitution set (>=, <=, >, <)
  bool_flip    and <-> or
  none_flip    is None <-> is not None
  arith_flip   + <-> -
new:
  off_by_one     integer literals / range() / subscript boundaries get +-1 variants
  wrong_variable swap a name reference for a same-ish local name visible in the span
  missing_not    insert / remove `not ` on an if / elif / while condition
contact families (2026-08-28, sol_answers/w7/a_contact_families.md build order —
these exist because gym gold-contact 12 vs fixed 4 was a family-EXPRESSIVENESS gap,
not a localization gap; each is AST-grounded, not regex-grounded):
  wrong_variable_used  replace exactly ONE Name load with an in-scope symbol whose
                       definition DOMINATES the use (ranked, one-node AST diff)
  missing_kwarg        insert `param=source` into ONE Call whose span-resolved
                       signature is missing that parameter
  missing_await        wrap ONE Call in `await`, only inside an async def and only
                       with span-local awaitable evidence for the callee
  boolean_operator     and <-> or on ONE BoolOp node (AST-precise; unlike the
                       line-level bool_flip it can target the 2nd operator)
  early_return_missing GUARD INSERTION: emit one `if <guard>: <action>` If subtree
                       above a legal statement boundary
attribute family (2026-08-29, measured gap: real-history ea639fc1::_build_chat_block,
gold `.message` -> `.content`, had NO family that could express it — the fixer's own
wrong-attribute arm only fires when a task-NAMED CLASS is findable in the corpus, and
`_build_chat_block`'s task text names none):
  wrong_attribute      swap ONE attribute load for another attribute that is
                       ACTUALLY ACCESSED elsewhere in the span (or, when the caller
                       passes them, the whole file). No class definition required —
                       the evidence is usage, not a schema.

Guard insertion is the one family whose edit is MULTI-LINE. It stays inside the
(line_idx, old_line, new_line) contract by encoding the inserted block plus the
original line as a single new_line containing embedded newlines: callers already
do `lines[li] = new_line` then `"\n".join(lines)`, so the apply/compile/cert path
needs no change. The 1B still never emits code — it picks a boundary, a guard
template, a symbol and an action from menus, and the assembler writes the subtree.

Class routing: cls names a single family -> only that family (plus, for cmp_flip,
its full substitution set). cls is None -> ALL families. An unrecognized cls falls
back to the four legacy families, exactly as the original code did.
"""
import ast
import re
import textwrap

from probes.real_repo_baseline import INJECTORS

import edit_sets

# ------------------------------------------------------------------ caps
# A line that offers more than this many corrections is not a menu, it is noise.
MAX_PER_LINE = 12
# Whole-function budget; the ranked order below puts class-matched families first
# so a truncation can never evict the gold fix for a class-routed enumeration.
MAX_PER_FUNCTION = 60
# Per-family per-line budgets (the two combinatorially explosive families).
MAX_OFF_BY_ONE_PER_LINE = 4
MAX_WRONG_VARIABLE_PER_LINE = 4
# contact-family budgets (per line unless noted)
MAX_WRONG_VARIABLE_USED_PER_LINE = 4
MAX_KWARG_PER_LINE = 3
MAX_AWAIT_PER_LINE = 2
MAX_BOOLEAN_OPERATOR_PER_LINE = 3
MAX_GUARDS_PER_FUNCTION = 8
# attribute family budgets
MAX_WRONG_ATTRIBUTE_PER_LINE = 4
# A file's attribute vocabulary is the family's whole evidence base; past this it
# is not evidence, it is a dictionary. Ranking decides which survive.
MAX_ATTRIBUTE_POOL = 40
# expressive-family budgets (real-19 autopsy fix #1)
MAX_LITERAL_PER_LINE = 4
MAX_LITERAL_POOL = 60
MAX_PASSTHROUGH_GUARDS_PER_FUNCTION = 8
MAX_CLAUSES_PER_LINE = 3

LEGACY_FAMILIES = ("cmp_flip", "bool_flip", "none_flip", "arith_flip")
NEW_FAMILIES = ("off_by_one", "wrong_variable", "missing_not")
# Appended LAST so the per-line and per-function caps can only ever evict a
# CONTACT candidate, never one of the seven families that were already measured.
CONTACT_FAMILIES = ("wrong_variable_used", "missing_kwarg", "missing_await",
                    "boolean_operator", "early_return_missing")
# Appended after the contact families for the same reason they were appended after
# the measured seven: a cap may only ever evict the NEWEST family's candidates.
ATTR_FAMILIES = ("wrong_attribute",)
# REAL-19 AUTOPSY FIX #1 — the expressiveness wall. Measured on the 19-task real
# board: exactly ONE gold fix was expressible by any enabled family, so even an
# oracle-perfect locate could not have scored above 1-2. These three families are
# the shapes the autopsy's own evidence names, and they append LAST for the same
# reason every earlier tranche did: a cap may only ever evict the newest family.
EXPRESSIVE_FAMILIES = ("literal_substitution", "none_passthrough_guard",
                       "clause_insertion")
# COORDINATED FAMILY (2026-08-29, w27 multi-line lane). The v3 real board is 10
# of 21 `multi_line_coordinated`, a shape NO single-line family can express: its
# candidate is an ordered SET of edits applied atomically, not one line. It lives
# in edit_sets.py and is appended LAST for the same reason every earlier tranche
# was — a cap may only ever evict the newest family. Structurally stronger than
# "appended last", in fact: edit-sets are added AFTER the per-function cap has
# already been spent, out of their own reserved budget, so they can neither
# evict nor reorder a single existing candidate. See the byte-identity guard in
# test_edit_sets_offline.py.
COORDINATED_FAMILIES = edit_sets.COORDINATED_FAMILIES
ALL_FAMILIES = (LEGACY_FAMILIES + NEW_FAMILIES + CONTACT_FAMILIES +
                ATTR_FAMILIES + EXPRESSIVE_FAMILIES + COORDINATED_FAMILIES)

_INT_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_RANGE_RE = re.compile(r"\brange\(\s*([^(),]+?)\s*\)")
_COND_RE = re.compile(r"^(\s*)(if|elif|while)\s+(.*?)(\s*):(\s*(?:#.*)?)$")


# ------------------------------------------------------------------ legacy
def _legacy_mutations(line, families):
    """The original four flip families. `families` is the set of legacy family
    names to run; cmp_flip additionally emits the full substitution set."""
    inj = {f.__name__.replace("flip_", ""): f for f in INJECTORS}
    fmap = {"arith_flip": inj["arith"], "cmp_flip": inj["cmp"],
            "bool_flip": inj["bool"], "none_flip": inj["none"]}
    out = []
    for name in LEGACY_FAMILIES:
        if name not in families:
            continue
        r = fmap[name](line)
        if r:
            out.append(r[0])
    # cmp needs the FULL substitution set — the pairwise flip is asymmetric
    # (bug >= -> < re-flips to >, never back to >=): offline showed 2/20 gold
    # fixes missing from the menu without this.
    if "cmp_flip" in families:
        ops = [" >= ", " <= ", " > ", " < "]
        for op in ops:
            if op in line:
                for rep in ops:
                    if rep != op:
                        out.append(line.replace(op, rep, 1))
                break
    return out


# ------------------------------------------------------------------ off_by_one
def _has_obo_context(line):
    """Bound the explosion: only lines with a comparison, a subscript, or a
    range/slice context are eligible."""
    if re.search(r"(<=|>=|==|!=|<|>)", line):
        return True
    if "[" in line and "]" in line:
        return True
    if re.search(r"\brange\s*\(", line):
        return True
    return False


def _off_by_one_mutations(line):
    if not _has_obo_context(line):
        return []
    out = []
    # 1) integer literals anywhere on the line -> +-1 (e.g. `> 5` -> `> 4` / `> 6`)
    for m in _INT_RE.finditer(line):
        try:
            val = int(m.group(1))
        except ValueError:
            continue
        for delta in (-1, 1):
            nv = val + delta
            if nv < 0:
                continue
            out.append(line[:m.start(1)] + str(nv) + line[m.end(1):])
    # 2) range(<expr>) with a non-literal bound -> range(<expr> + 1) / (- 1)
    for m in _RANGE_RE.finditer(line):
        arg = m.group(1)
        if _INT_RE.search(arg) or re.search(r"[+\-]", arg):
            continue  # literal bounds already covered above / already offset
        for delta in ("+ 1", "- 1"):
            out.append(line[:m.start(1)] + f"{arg} {delta}" + line[m.end(1):])
    # 3) subscript boundaries: `x[i - 1]` -> `x[i]`, `x[i]` -> `x[i - 1]` / `[i + 1]`
    for m in re.finditer(r"\[\s*([A-Za-z_]\w*)\s*([+\-])\s*1\s*\]", line):
        out.append(line[:m.start()] + f"[{m.group(1)}]" + line[m.end():])
    for m in re.finditer(r"\[\s*([A-Za-z_]\w*)\s*\]", line):
        for op in ("-", "+"):
            out.append(line[:m.start()] + f"[{m.group(1)} {op} 1]" + line[m.end():])
    return _cap(out, MAX_OFF_BY_ONE_PER_LINE, line)


# ------------------------------------------------------------------ wrong_variable
def collect_span_names(span_lines):
    """Function-local names visible in the span: assignment targets, arguments,
    for-targets and every loaded name. Parses the joined span; if the span is a
    bare body it is re-parsed wrapped in a dummy def."""
    text = textwrap.dedent("\n".join(span_lines))
    tree = None
    for source in (text, "def _dummy():\n" + textwrap.indent(text, "    ")):
        try:
            tree = ast.parse(source)
            break
        except SyntaxError:
            continue
    if tree is None:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    names.discard("_dummy")
    return {n for n in names if len(n) >= 2}


def _same_ish(a, b):
    """Same-ish: shared prefix or suffix of >= 4 chars, or differing by exactly
    one underscore-separated token (start_idx/end_idx, user_id/learner_id)."""
    if a == b:
        return False
    n = min(len(a), len(b))
    pref = 0
    while pref < n and a[pref] == b[pref]:
        pref += 1
    if pref >= 4:
        return True
    suf = 0
    while suf < n and a[-1 - suf] == b[-1 - suf]:
        suf += 1
    if suf >= 4:
        return True
    ta, tb = a.split("_"), b.split("_")
    if len(ta) == len(tb) and len(ta) > 1:
        if sum(1 for x, y in zip(ta, tb) if x != y) == 1:
            return True
    return False


def _wrong_variable_mutations(line, names):
    if not names:
        return []
    out = []
    present = sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", line))
    for name in present:
        alts = sorted(o for o in names if _same_ish(name, o))
        if not alts:
            continue
        # per-occurrence so a swap can target the 2nd reference on the line
        for m in re.finditer(rf"\b{re.escape(name)}\b", line):
            if m.start() > 0 and line[m.start() - 1] == ".":
                continue  # attribute access, not a local name reference
            for alt in alts:
                out.append(line[:m.start()] + alt + line[m.end():])
    return _cap(out, MAX_WRONG_VARIABLE_PER_LINE, line)


# ------------------------------------------------------------------ missing_not
def _missing_not_mutations(line):
    m = _COND_RE.match(line)
    if not m:
        return []
    indent, kw, cond, pad, tail = m.groups()

    def rebuild(new_cond):
        return f"{indent}{kw} {new_cond}{pad}:{tail}"

    stripped = cond.strip()
    if stripped.startswith("not "):
        return [rebuild(stripped[4:].strip())]
    if not stripped:
        return []
    return [rebuild(f"not {stripped}")]


# =====================================================================
# CONTACT FAMILIES — shared AST fact layer
# =====================================================================
# Every contact family is generated from FACTS about the span (definitions and
# their line offsets, resolved signatures, async boundaries, BoolOp positions),
# never from a regex over the line. That is what lets the witness re-derive the
# same facts INDEPENDENTLY and kill a candidate the generator (or a model)
# produced from a bad premise.

class SpanFacts:
    """Parsed, offset-indexed facts about one function span.

    `pads[i]` is the number of leading characters textwrap.dedent stripped from
    span line i, so an AST col_offset (which indexes the DEDENTED text) maps back
    onto the original line by adding pads[i].
    """

    def __init__(self, span_lines, tree, dd_lines, pads):
        self.span_lines = list(span_lines)
        self.tree = tree
        self.dd_lines = dd_lines
        self.pads = pads
        self.defs = {}          # name -> FunctionDef/AsyncFunctionDef node
        self.async_defs = set()  # names defined with `async def`
        self.sym_def_off = {}   # symbol -> EARLIEST offset at which it is bound
        self.enclosing = {}     # stmt offset -> enclosing function node
        self.awaited = set()    # callee names already awaited somewhere in the span

    def line_of(self, node):
        """0-based span offset of a node, or None when it spans several lines."""
        lineno = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", lineno)
        if lineno is None or end != lineno:
            return None
        off = lineno - 1
        return off if 0 <= off < len(self.span_lines) else None

    def segment(self, node):
        """(offset, start_col, end_col) of a single-line node in the ORIGINAL line."""
        off = self.line_of(node)
        if off is None:
            return None
        c0 = getattr(node, "col_offset", None)
        c1 = getattr(node, "end_col_offset", None)
        if c0 is None or c1 is None:
            return None
        pad = self.pads[off]
        return off, c0 + pad, c1 + pad

    def defined_before(self, name, off):
        """Is `name` bound at a line strictly above `off` (arguments count as
        bound at the def line)? This is the dominance half of the witness."""
        d = self.sym_def_off.get(name)
        return d is not None and d < off


def _bind(facts, name, off):
    if not name:
        return
    prev = facts.sym_def_off.get(name)
    if prev is None or off < prev:
        facts.sym_def_off[name] = off


def span_facts(span_lines):
    """Parse the span and collect the fact table, or None when it does not parse.

    Deliberately strict: a span we cannot parse yields NO contact candidates at
    all rather than regex guesses. Fail-closed on generation is safe — the seven
    existing families still enumerate.
    """
    if not span_lines:
        return None
    text = textwrap.dedent("\n".join(span_lines))
    dd_lines = text.split("\n")
    if len(dd_lines) != len(span_lines):
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    pads = [len(o) - len(d) for o, d in zip(span_lines, dd_lines)]
    if any(p < 0 for p in pads):
        return None
    facts = SpanFacts(span_lines, tree, dd_lines, pads)

    def walk(node, fn):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                off = getattr(child, "lineno", 1) - 1
                if 0 <= off < len(span_lines):
                    facts.enclosing[off] = fn
            inner = fn
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                off = child.lineno - 1
                facts.defs[child.name] = child
                if isinstance(child, ast.AsyncFunctionDef):
                    facts.async_defs.add(child.name)
                _bind(facts, child.name, off)
                for a in (list(child.args.posonlyargs) + list(child.args.args) +
                          list(child.args.kwonlyargs) +
                          [child.args.vararg, child.args.kwarg]):
                    if a is not None:
                        _bind(facts, a.arg, off)
                inner = child
            elif isinstance(child, ast.Assign):
                for t in child.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            _bind(facts, n.id, child.lineno - 1)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(child.target, ast.Name):
                    _bind(facts, child.target.id, child.lineno - 1)
            elif isinstance(child, ast.For):
                for n in ast.walk(child.target):
                    if isinstance(n, ast.Name):
                        _bind(facts, n.id, child.lineno - 1)
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if item.optional_vars is not None:
                        for n in ast.walk(item.optional_vars):
                            if isinstance(n, ast.Name):
                                _bind(facts, n.id, child.lineno - 1)
            elif isinstance(child, ast.Await) and isinstance(child.value, ast.Call):
                cname = _callee_name(child.value)
                if cname:
                    facts.awaited.add(cname)
            walk(child, inner)

    walk(facts.tree, None)
    return facts


def _callee_name(call):
    """Dotted callee text for a Call, or None."""
    fn = call.func
    parts = []
    while isinstance(fn, ast.Attribute):
        parts.append(fn.attr)
        fn = fn.value
    if not isinstance(fn, ast.Name):
        return None
    parts.append(fn.id)
    return ".".join(reversed(parts))


def _splice(line, c0, c1, replacement):
    return line[:c0] + replacement + line[c1:]


def _rank_symbols(target, candidates):
    """Deterministic dataflow-similarity ranking: same-ish first (shared >=4-char
    prefix/suffix, or one differing underscore token), then by shared-prefix
    length, then alphabetically so the menu order never wobbles."""
    def key(name):
        n = min(len(target), len(name))
        pref = 0
        while pref < n and target[pref] == name[pref]:
            pref += 1
        return (0 if _same_ish(target, name) else 1, -pref, name)
    return sorted(candidates, key=key)


# ------------------------------------------------- (1) wrong_variable_used
def _wrong_variable_used(facts):
    """Replace exactly ONE Name load with an in-scope symbol whose binding
    DOMINATES the use. One-node AST diff by construction."""
    sites = {}                              # line offset -> [(c0, c1, ranked alts)]
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        seg = facts.segment(node)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        if line[c0:c1] != node.id:          # column mapping disagrees -> skip
            continue
        cands = _rank_symbols(node.id, [
            n for n in facts.sym_def_off
            if n != node.id and n not in facts.defs
            and facts.defined_before(n, off)])
        if cands:
            sites.setdefault(off, []).append((c0, c1, cands))
    # ROUND-ROBIN across the line's target sites, not first-come. A line with the
    # same name twice (`items[start_idx:start_idx]`) must be able to reach the 2nd
    # occurrence — first-come ordering burned the whole per-line budget on the 1st
    # and the gold fix never made the menu.
    out = []
    for off in sorted(sites):
        line = facts.span_lines[off]
        targets = sorted(sites[off], key=lambda t: t[0])
        emitted, rank = 0, 0
        while emitted < MAX_WRONG_VARIABLE_USED_PER_LINE:
            progressed = False
            for c0, c1, alts in targets:
                if rank >= len(alts):
                    continue
                progressed = True
                out.append((off, line, _splice(line, c0, c1, alts[rank])))
                emitted += 1
                if emitted >= MAX_WRONG_VARIABLE_USED_PER_LINE:
                    break
            if not progressed:
                break
            rank += 1
    return out


# ------------------------------------------------- (2) missing_kwarg
def _signature_params(fn):
    """(required, all) parameter names of a def, keyword-passable only."""
    args = fn.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    n_req = len(positional) - len(defaults)
    names = [a.arg for a in positional if a.arg not in ("self", "cls")]
    kwonly = [a.arg for a in args.kwonlyargs]
    required = [a.arg for a in positional[:n_req] if a.arg not in ("self", "cls")]
    required += [a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults)
                 if d is None]
    return required, names + kwonly


def _missing_kwarg(facts):
    """Insert `param=source` into ONE Call whose signature (resolved from a def in
    the same span) is missing that parameter and whose source symbol is in scope."""
    out = []
    per_line = {}
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        fn = facts.defs.get(name)
        if fn is None:
            continue
        seg = facts.segment(node)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        if not line[c0:c1].endswith(")"):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args) or \
                any(k.arg is None for k in node.keywords):
            continue                        # *args/**kwargs: binding unresolvable
        required, allowed = _signature_params(fn)
        supplied = set(allowed[:len(node.args)]) | {k.arg for k in node.keywords}
        missing = [p for p in allowed if p not in supplied]
        has_args = bool(node.args or node.keywords)
        for param in missing:
            # source symbol: the same name in scope, else the closest same-ish one
            sources = [param] if facts.defined_before(param, off) else []
            if not sources:
                sources = [s for s in _rank_symbols(param, [
                    n for n in facts.sym_def_off
                    if facts.defined_before(n, off) and _same_ish(param, n)])][:1]
            for src in sources:
                if per_line.get(off, 0) >= MAX_KWARG_PER_LINE:
                    break
                per_line[off] = per_line.get(off, 0) + 1
                ins = (", " if has_args else "") + f"{param}={src}"
                new_call = line[c0:c1 - 1] + ins + ")"
                out.append((off, line, _splice(line, c0, c1, new_call)))
    return out


# ------------------------------------------------- (3) missing_await
def _missing_await(facts):
    """Wrap ONE Call in `await`. Fires ONLY inside an `async def`, and only with
    span-local awaitable evidence: the callee is an `async def` in the span, or
    the very same callee is already awaited on another line (the copy-paste
    missing-await shape). Never in a sync function."""
    out = []
    per_line = {}

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            inner = child if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.Call) and isinstance(
                    fn, ast.AsyncFunctionDef):
                name = _callee_name(child)
                awaitable = name in facts.async_defs or name in facts.awaited
                seg = facts.segment(child)
                if awaitable and seg is not None and not _already_awaited(
                        facts, child):
                    off, c0, c1 = seg
                    line = facts.span_lines[off]
                    if per_line.get(off, 0) < MAX_AWAIT_PER_LINE:
                        per_line[off] = per_line.get(off, 0) + 1
                        out.append((off, line,
                                    _splice(line, c0, c1,
                                            "await " + line[c0:c1])))
            visit(child, inner)

    visit(facts.tree, None)
    return out


def _already_awaited(facts, call):
    for node in ast.walk(facts.tree):
        if isinstance(node, ast.Await) and node.value is call:
            return True
    return False


# ------------------------------------------------- (4) boolean_operator
def _boolean_operator(facts):
    """and <-> or on ONE BoolOp node. AST-precise, so unlike the line-level
    bool_flip it can target the second operator on a line."""
    out = []
    per_line = {}
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.BoolOp):
            continue
        seg = facts.segment(node)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        src = line[c0:c1]
        old_kw = "and" if isinstance(node.op, ast.And) else "or"
        new_kw = "or" if old_kw == "and" else "and"
        swapped = re.sub(rf"(?<![\w.]){old_kw}(?![\w])", new_kw, src)
        if swapped == src:
            continue
        if per_line.get(off, 0) >= MAX_BOOLEAN_OPERATOR_PER_LINE:
            continue
        per_line[off] = per_line.get(off, 0) + 1
        out.append((off, line, _splice(line, c0, c1, swapped)))
    return out


# ------------------------------------------------- (5) early_return_missing
# The typed tuple the 1B fills in:
#   (boundary_offset, guard_template_id, symbol, action_template_id)
GUARD_TEMPLATES = (("is_none", "{sym} is None"), ("falsy", "not {sym}"))
ACTION_TEMPLATES = (("return_none", "return None"), ("bare_return", "return"))


def _returns_a_value(fn):
    return any(isinstance(n, ast.Return) and n.value is not None
               for n in ast.walk(fn))


def _early_return_missing(facts):
    """GUARD INSERTION. Emits one `If` subtree above a legal statement boundary.

    Gates at generation time (the witness re-checks all of them independently):
      * boundary must be a statement START inside a function body
      * the guarded symbol must be BOUND STRICTLY ABOVE the boundary
      * the action must be return-compatible with the enclosing function
      * an equivalent guard must not already exist in the span
    """
    out = []
    existing = {re.sub(r"\s+", " ", l.strip()) for l in facts.span_lines}
    # A docstring is not a statement a guard can meaningfully precede: inserting
    # above it is behaviour-inert (it survives every mutation-kill oracle) AND it
    # displaces the docstring out of position. Excluded at the boundary menu.
    docstring_offs = set()
    for node in ast.walk(facts.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            first = (node.body or [None])[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstring_offs.add(first.lineno - 1)
    for off, fn in sorted(facts.enclosing.items()):
        if fn is None or off == fn.lineno - 1 or off in docstring_offs:
            continue                         # the def line is not a boundary
        line = facts.span_lines[off]
        indent = line[:len(line) - len(line.lstrip())]
        if not line.strip() or line.lstrip().startswith(("#", "else", "elif",
                                                         "except", "finally")):
            continue
        symbols = sorted(n for n in facts.sym_def_off
                         if facts.defined_before(n, off) and n not in facts.defs)
        if not symbols:
            continue
        valued = _returns_a_value(fn)
        actions = [a for a in ACTION_TEMPLATES
                   if not (a[0] == "bare_return" and valued)]
        for sym in symbols:
            for _gid, guard_tpl in GUARD_TEMPLATES:
                guard = guard_tpl.format(sym=sym)
                head = f"{indent}if {guard}:"
                if re.sub(r"\s+", " ", head.strip()) in existing:
                    continue                 # duplicate guard
                for _aid, action in actions:
                    if len(out) >= MAX_GUARDS_PER_FUNCTION:
                        return out
                    block = f"{head}\n{indent}    {action}\n{line}"
                    out.append((off, line, block))
    return out


CONTACT_GENERATORS = {
    "wrong_variable_used": _wrong_variable_used,
    "missing_kwarg": _missing_kwarg,
    "missing_await": _missing_await,
    "boolean_operator": _boolean_operator,
    "early_return_missing": _early_return_missing,
}


# ------------------------------------------- none_flip: f-string interiors (E52)
def _none_fstring_mutations(facts):
    """`is None` <-> `is not None` flips inside F-STRING REPLACEMENT FIELDS.

    The legacy none_flip is a TEXT substitution over the raw line: it rewrites the
    first ` is None` it sees anywhere, so a line holding two replacement fields
    only ever offers one candidate, and a line whose PROSE happens to say "is
    None" offers a bogus one. Both are the same defect — the family never looked
    at structure.

    This generator does: every `Compare` whose single operator is `Is`/`IsNot`
    against the `None` constant, reached through a `JoinedStr` -> `FormattedValue`
    interior, yields one flip AT ITS OWN SPAN. Prose segments of the f-string are
    never inside a Compare's span, so they are untouched by construction, and N
    fields on one line yield N independent candidates.

    Fail-closed on position drift: older CPythons give nodes inside an f-string
    approximate offsets, so the slice is VERIFIED to end with the operator text
    the AST said was there before any candidate is emitted. A span that does not
    verify yields nothing rather than a corrupted line.
    """
    out = []
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for fv in ast.walk(node):
            if not isinstance(fv, ast.FormattedValue):
                continue
            for cmp_node in ast.walk(fv.value):
                if not isinstance(cmp_node, ast.Compare) or len(cmp_node.ops) != 1:
                    continue
                op = cmp_node.ops[0]
                if not isinstance(op, (ast.Is, ast.IsNot)):
                    continue
                right = cmp_node.comparators[0]
                if not (isinstance(right, ast.Constant) and right.value is None):
                    continue
                seg = facts.segment(cmp_node)
                if seg is None:
                    continue
                off, c0, c1 = seg
                line = facts.span_lines[off]
                if not (0 <= c0 < c1 <= len(line)):
                    continue
                text = line[c0:c1]
                was = " is not None" if isinstance(op, ast.IsNot) else " is None"
                now = " is None" if isinstance(op, ast.IsNot) else " is not None"
                if not text.endswith(was):
                    continue            # offsets did not verify: emit nothing
                out.append((off, line, line[:c0] + text[:-len(was)] + now + line[c1:]))
    return out


# ------------------------------------------------- (6) wrong_attribute
def _receiver_root(node):
    """A stable identity for the object an Attribute is read off.

    `msg.message` -> "msg"; `response.choices[0].message.content` -> "message"
    (the nearest attribute in the chain). Used only for RANKING, so an imperfect
    identity costs menu order, never correctness.
    """
    value = node.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _call_func_attributes(tree):
    """Attribute nodes that are the callee of a Call — i.e. METHODS.

    `.strip()`/`.replace()`/`.all()` are behaviour of the type, not fields of the
    record, and swapping one for another produces noise rather than a plausible
    wrong-attribute bug. Excluded from both the pool and the target set.
    """
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.add(id(node.func))
    return out


def collect_attribute_pool(source_lines, span_names=()):
    """attr name -> {"recv": set of receiver identities, "count": times read}.

    This is the family's entire premise: an attribute is a legal correction ONLY
    because some other line in the same code actually reads it off something. No
    class definition, no schema, no task-named type — pure observed usage, which
    is what makes the family fire on `_build_chat_block`, whose task text names no
    class at all. Unparseable input yields an EMPTY pool (fail closed).
    """
    pool = {}
    if not source_lines:
        return pool
    text = textwrap.dedent("\n".join(source_lines))
    tree = None
    for source in (text, "def _dummy():\n" + textwrap.indent(text, "    ")):
        try:
            tree = ast.parse(source)
            break
        except SyntaxError:
            continue
    if tree is None:
        return pool
    methods = _call_func_attributes(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in methods:
            continue
        attr = node.attr
        if attr.startswith("_") or attr.isupper() or len(attr) < 2:
            continue                       # dunders/privates/class constants
        entry = pool.setdefault(attr, {"recv": set(), "count": 0})
        entry["recv"].add(_receiver_root(node))
        entry["count"] += 1
    if len(pool) > MAX_ATTRIBUTE_POOL:
        # Truncating the pool must not drop the strongest evidence: keep the
        # span-named attributes first, then the most-read ones.
        named = set(span_names or ())
        keep = sorted(pool, key=lambda a: (a not in named, -pool[a]["count"], a))
        pool = {a: pool[a] for a in keep[:MAX_ATTRIBUTE_POOL]}
    return pool


def _rank_attributes(target, receiver, pool, span_names=()):
    """Deterministic order, strongest evidence first:

    0. the attribute's name is ALSO a plain identifier in the span. This is the
       real-history shape: `content = msg.message[:120]` — the code already tells
       you what the value is called, and the field it should have come from has
       the same name. Without this tier the correct swap sits behind every other
       field the file happens to mention and never reaches the menu.
    1. attributes observed on the SAME receiver identity
    2. same-ish names, then shared-prefix length
    3. most-read in the source, then alphabetical (stable order, always)
    """
    named = set(span_names or ())

    def key(name):
        n = min(len(target), len(name))
        pref = 0
        while pref < n and target[pref] == name[pref]:
            pref += 1
        return (name not in named,
                0 if (receiver is not None
                      and receiver in pool[name]["recv"]) else 1,
                0 if _same_ish(target, name) else 1,
                -pref, -pool[name]["count"], name)
    return sorted((a for a in pool if a != target), key=key)


def _wrong_attribute(facts, pool, span_names=()):
    """Swap ONE attribute LOAD for another attribute observed in the code.

    Deliberately NOT gated on a class definition being present: that gate is
    exactly why the real-history `.message` -> `.content` fix had no expressible
    candidate. The column-slice is VERIFIED against the source text before any
    candidate is emitted, so an offset drift yields nothing rather than a
    corrupted line (same fail-closed rule as the f-string none_flip arm).
    """
    if not pool:
        return []
    methods = _call_func_attributes(facts.tree)
    sites = {}
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.Attribute) or id(node) in methods:
            continue
        if not isinstance(node.ctx, ast.Load):
            continue                        # never rewrite a STORE target
        seg = facts.segment(node)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        if not (0 <= c0 < c1 <= len(line)):
            continue
        text = line[c0:c1]
        suffix = "." + node.attr
        if not text.endswith(suffix):
            continue                        # offsets did not verify
        alts = _rank_attributes(node.attr, _receiver_root(node), pool,
                                span_names)
        if alts:
            sites.setdefault(off, []).append((c0, c1, text, suffix, alts))
    out = []
    for off in sorted(sites):
        line = facts.span_lines[off]
        targets = sorted(sites[off], key=lambda t: t[0])
        emitted, rank = 0, 0
        # ROUND-ROBIN across this line's attribute sites, same rule as
        # _wrong_variable_used: a line with two attribute reads must be able to
        # reach the second one instead of spending the budget on the first.
        while emitted < MAX_WRONG_ATTRIBUTE_PER_LINE:
            progressed = False
            for c0, c1, text, suffix, alts in targets:
                if rank >= len(alts):
                    continue
                progressed = True
                new_text = text[:-len(suffix)] + "." + alts[rank]
                out.append((off, line, _splice(line, c0, c1, new_text)))
                emitted += 1
                if emitted >= MAX_WRONG_ATTRIBUTE_PER_LINE:
                    break
            if not progressed:
                break
            rank += 1
    return out


# =========================================================================
# EXPRESSIVE FAMILIES (real-19 autopsy fix #1)
#
# The autopsy measured the binding constraint on the real board: of 19 real
# fixes exactly ONE was a substitution any enabled family could express. Locate
# quality is irrelevant to a fix that has no candidate. These three generators
# are ATOMS-SHAPED like every family before them — the menu offers concrete
# candidate values / clauses MINED FROM CONTEXT and the model picks one; nothing
# here asks a model to write code.
# =========================================================================

_IDENT_LITERAL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[.\-][A-Za-z_][A-Za-z0-9_]*)*$")
_CAMEL_SEG_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
# String literals mined out of a file's raw text: the pool for a key rename.
_FILE_LITERAL_RE = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_.\-]{2,})['"]""")


def _docstring_offsets(facts):
    """Span offsets holding a docstring's FIRST line. A docstring is prose: no
    family may mutate it and no guard may be inserted above it."""
    offs = set()
    for node in ast.walk(facts.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            first = (node.body or [None])[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                offs.add(first.lineno - 1)
    return offs


def _docstring_nodes(tree):
    """The Constant nodes that ARE docstrings, by identity."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            first = (node.body or [None])[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _segment_drop_variants(s):
    """Every string obtained by deleting ONE snake_case or camelCase segment.

    This is the morphological half of `literal_substitution`, and it is the half
    that carries its own evidence: `globalBrainiacMemoryWritten` ->
    `globalMemoryWritten` is derivable from the literal alone, with no pool and
    no corpus. Order is left-to-right and deterministic.
    """
    # INTERIOR SEGMENTS FIRST. A name's head and tail carry its identity (what
    # it is, and what kind of thing it is); the word that turns out to be one too
    # many is an interior qualifier. Emitting interior drops first is what keeps
    # a long function's per-function round-robin from evicting the candidate at
    # rank 1 — measured on `globalBrainiacMemoryWritten`, where the gold
    # `globalMemoryWritten` is the first interior drop and the head/tail drops
    # (`BrainiacMemoryWritten`, `globalBrainiacMemory`) are noise.
    def ordered(n):
        return list(range(1, n - 1)) + ([0, n - 1] if n > 1 else [])

    out = []
    if "_" in s:
        parts = s.split("_")
        if len(parts) > 1:
            for i in ordered(len(parts)):
                cand = "_".join(parts[:i] + parts[i + 1:])
                if cand and cand != s and cand not in out:
                    out.append(cand)
    segs = [m.span() for m in _CAMEL_SEG_RE.finditer(s)]
    if len(segs) > 1:
        for i in ordered(len(segs)):
            a, b = segs[i]
            cand = s[:a] + s[b:]
            if cand and cand != s and cand not in out:
                out.append(cand)
    return out


def collect_literal_pool(file_lines):
    """Identifier-shaped string literals used ANYWHERE in the supplied source.

    The evidence base for a key/constant rename. Deliberately a raw-text scan:
    the pool must include literals from parts of the file that do not parse in
    isolation, and a rename's answer is a token the codebase already uses.
    """
    if not file_lines:
        return []
    text = "\n".join(file_lines)
    seen, out = set(), []
    for tok in _FILE_LITERAL_RE.findall(text):
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= MAX_LITERAL_POOL:
            break
    return out


def _rank_literals(target, candidates):
    """Closest-first by character similarity, ties broken alphabetically so the
    menu order never wobbles across runs."""
    import difflib

    def key(c):
        return (-difflib.SequenceMatcher(None, target, c).ratio(), c)
    return sorted(candidates, key=key)


def _literal_substitution(facts, pool=()):
    """Swap ONE identifier-shaped string constant for another.

    The real-board shapes: a dict KEY that must be spelled the way the consumer
    reads it (`"contract_slug"` -> `"contract_hint"`), and a receipt field whose
    name carries one word too many (`"globalBrainiacMemoryWritten"` ->
    `"globalMemoryWritten"`). Neither is a mutation of any operator, so no
    existing family can reach them.

    Candidate order: SELF-DERIVED segment-drop variants first (they need no
    pool and cannot be wrong about provenance), then pool literals the file
    already uses, closest-first. Docstrings and f-string interiors are excluded
    by construction; every splice is verified against the original quoting
    before it is emitted, so a column-mapping disagreement yields nothing.
    """
    doc_ids = _docstring_nodes(facts.tree)
    sites = {}
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in doc_ids:
            continue
        val = node.value
        if len(val) < 3 or not _IDENT_LITERAL_RE.match(val):
            continue
        seg = facts.segment(node)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        text = line[c0:c1]
        # FAIL-CLOSED on quoting: the slice must be exactly this literal, quotes
        # included. An f-string part or a drifted offset never satisfies this.
        if len(text) < 2 or text[0] not in "'\"" or text[-1] != text[0] \
                or text[1:-1] != val:
            continue
        alts = _segment_drop_variants(val)
        alts += [c for c in _rank_literals(val, [p for p in pool if p != val])
                 if c not in alts]
        if alts:
            sites.setdefault(off, []).append((c0, c1, text[0], alts))
    out = []
    for off in sorted(sites):
        line = facts.span_lines[off]
        targets = sorted(sites[off], key=lambda t: t[0])
        emitted, rank = 0, 0
        while emitted < MAX_LITERAL_PER_LINE:
            progressed = False
            for c0, c1, q, alts in targets:
                if rank >= len(alts):
                    continue
                progressed = True
                out.append((off, line, _splice(line, c0, c1, q + alts[rank] + q)))
                emitted += 1
                if emitted >= MAX_LITERAL_PER_LINE:
                    break
            if not progressed:
                break
            rank += 1
    return out


def _loop_body_offsets(tree):
    """Offsets of statements that sit DIRECTLY in a for/while body — the only
    boundaries where `continue` is legal."""
    offs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            for stmt in list(node.body) + list(node.orelse):
                offs.add(stmt.lineno - 1)
    return offs


def _none_passthrough_guard(facts):
    """Insert `if <sym> is None: continue` above a legal loop-body boundary.

    Distinct from `early_return_missing`, which guards by RETURNING: a
    passthrough guard says "this iteration has nothing to check", which is the
    recurring real-world shape behind `_validate_fields` (an explicit null is
    the same statement as an absent optional key). `continue` is emitted ONLY at
    a boundary directly inside a loop, so the candidate always compiles.

    Every gate `early_return_missing` applies is applied here too: the symbol
    must be bound strictly above the boundary, the boundary is never a def line
    or a docstring, and an equivalent guard already present in the span
    suppresses the duplicate.
    """
    existing = {re.sub(r"\s+", " ", l.strip()) for l in facts.span_lines}
    docstring_offs = _docstring_offsets(facts)
    loop_offs = _loop_body_offsets(facts.tree)
    sites = []                              # [(off, line, [candidate block, ...])]
    for off, fn in sorted(facts.enclosing.items()):
        if fn is None or off == fn.lineno - 1 or off in docstring_offs:
            continue
        if off not in loop_offs:
            continue                       # `continue` outside a loop is a SyntaxError
        line = facts.span_lines[off]
        if not line.strip() or line.lstrip().startswith(("#", "else", "elif",
                                                         "except", "finally")):
            continue
        indent = line[:len(line) - len(line.lstrip())]
        # CLOSEST BINDING FIRST. The value a statement is about is overwhelmingly
        # the one bound just above it; alphabetical order put `config` ahead of
        # `value` and the per-function cap then spent the whole budget on the
        # first boundary, so the gold guard was never enumerated at all.
        symbols = sorted((n for n in facts.sym_def_off
                          if facts.defined_before(n, off) and n not in facts.defs),
                         key=lambda n: (-facts.sym_def_off[n], n))
        blocks = []
        for sym in symbols:
            head = f"{indent}if {sym} is None:"
            if re.sub(r"\s+", " ", head.strip()) in existing:
                continue                   # duplicate guard
            blocks.append(f"{head}\n{indent}    continue\n{line}")
        if blocks:
            sites.append((off, line, blocks))
    # ROUND-ROBIN across boundaries: every boundary offers its closest-bound
    # symbol before any boundary offers its second.
    out, rank = [], 0
    while len(out) < MAX_PASSTHROUGH_GUARDS_PER_FUNCTION:
        progressed = False
        for off, line, blocks in sites:
            if rank >= len(blocks):
                continue
            progressed = True
            out.append((off, line, blocks[rank]))
            if len(out) >= MAX_PASSTHROUGH_GUARDS_PER_FUNCTION:
                break
        if not progressed:
            break
        rank += 1
    return out


_TYPE_NAMES = {str: "str", int: "int", float: "float", bytes: "bytes"}
_CHAIN_CALLEES = frozenset(("filter", "where", "having"))


def _homogeneous_constant_type(node):
    """The single python type of a Set/List/Tuple of constants, or None."""
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return None
    elts = list(node.elts)
    if not elts:
        return None
    types = set()
    for e in elts:
        if not isinstance(e, ast.Constant) or isinstance(e.value, bool):
            return None
        types.add(type(e.value))
    if len(types) != 1:
        return None
    return _TYPE_NAMES.get(next(iter(types)))


def _clause_insertion(facts, file_lines=None):
    """Add ONE clause to an existing condition or filter chain.

    Two shapes, both mined from context, never invented:

      TYPE-GUARD DISJUNCT.  `x not in {"a", "b"}` is a TypeError waiting for an
      unhashable `x`; the element type of the literal set is the evidence for
      which type to guard, so `not isinstance(x, str) or <cond>` is derived from
      the condition itself. The `in` polarity gets the conjunct form.

      CHAIN CLAUSE.  A `.filter(...)` / `.where(...)` argument list gains one
      clause that a SIBLING call to the same method elsewhere in the file
      already uses. This is the `_pipeline_queue_stall_checks` shape: the repair
      is "mirror the query beside you", and the clause to mirror is literally
      written in the neighbouring call.
    """
    out = []
    # ---- shape 1: type-guard disjunct on a membership test
    for node in ast.walk(facts.tree):
        if not isinstance(node, (ast.If, ast.While)):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.left, ast.Name):
            continue
        tname = _homogeneous_constant_type(test.comparators[0])
        if tname is None:
            continue
        seg = facts.segment(test)
        if seg is None:
            continue
        off, c0, c1 = seg
        line = facts.span_lines[off]
        cond = line[c0:c1]
        if not cond.strip():
            continue
        sym = test.left.id
        if f"isinstance({sym}," in line:
            continue                       # the guard is already there
        if isinstance(test.ops[0], ast.NotIn):
            new_cond = f"not isinstance({sym}, {tname}) or {cond}"
        elif isinstance(test.ops[0], ast.In):
            new_cond = f"isinstance({sym}, {tname}) and {cond}"
        else:
            continue
        out.append((off, line, _splice(line, c0, c1, new_cond)))
    # ---- shape 2: mirror one clause from a sibling chain call
    out += _chain_clause_insertion(facts, file_lines)
    return out


def _chain_clause_source(node, lines):
    """Source text of a single-line call argument, or None."""
    lineno, end = getattr(node, "lineno", None), getattr(node, "end_lineno", None)
    if lineno is None or end != lineno or not (0 < lineno <= len(lines)):
        return None
    c0, c1 = getattr(node, "col_offset", None), getattr(node, "end_col_offset", None)
    if c0 is None or c1 is None:
        return None
    return lines[lineno - 1][c0:c1].strip()


def _sibling_chain_clauses(file_lines):
    """{callee name -> [clause source, ...]} over every chain call in the file.

    The evidence base for "mirror the query beside you". Fails CLOSED: a file
    that does not parse contributes nothing rather than regex guesses.
    """
    if not file_lines:
        return {}
    try:
        tree = ast.parse("\n".join(file_lines))
    except SyntaxError:
        return {}
    pool = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name not in _CHAIN_CALLEES:
            continue
        for arg in node.args:
            src = _chain_clause_source(arg, file_lines)
            if src and len(src) > 2:
                bucket = pool.setdefault(name, [])
                if src not in bucket:
                    bucket.append(src)
    return pool


def _chain_clause_insertion(facts, file_lines=None):
    """Append one sibling-mined clause to a chain call's argument list."""
    pool = _sibling_chain_clauses(file_lines)
    if not pool:
        return []
    out = []
    for node in ast.walk(facts.tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name not in pool or not node.args:
            continue
        own = {s for s in (_chain_clause_source(a, facts.span_lines)
                           for a in node.args) if s}
        last = node.args[-1]
        off = facts.line_of(last)
        if off is None:
            continue
        line = facts.span_lines[off]
        indent = line[:len(line) - len(line.lstrip())]
        emitted = 0
        for clause in pool[name]:
            if clause in own or clause in "\n".join(facts.span_lines):
                continue
            base = line if line.rstrip().endswith(",") else line.rstrip() + ","
            out.append((off, line, f"{base}\n{indent}{clause},"))
            emitted += 1
            if emitted >= MAX_CLAUSES_PER_LINE:
                break
    return out


EXPRESSIVE_GENERATORS = {
    "literal_substitution": _literal_substitution,
    "none_passthrough_guard": _none_passthrough_guard,
    "clause_insertion": _clause_insertion,
}


# ------------------------------------------------------------------ helpers
def _cap(candidates, limit, old):
    seen, out = set(), []
    for c in candidates:
        if c == old or c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _resolve_families(cls):
    if cls is None:
        return list(ALL_FAMILIES)
    if cls in ALL_FAMILIES:
        return [cls]
    # Unrecognized class: the original code fell back to all four legacy flips.
    return list(LEGACY_FAMILIES)


# ------------------------------------------------------------------ entry point
def enumerate_mutations(span_lines, start_idx, cls, file_lines=None,
                        providers=None):
    """Closed candidate set: class-matched single-line mutations over the function.

    Applying an injector to the BUGGED line of that class yields a candidate that
    (for flips) restores the original — the fix is in this set by construction.

    `file_lines` is OPTIONAL context (the whole source file the span came from).
    `wrong_attribute` reads it to widen its evidence base from "attributes used
    in this function" to "attributes used in this file", and the coordinated
    family's census tranche reads it for scope, peer and exception evidence.

    `providers` is an OPTIONAL `edit_sets.ProviderIndex` over the whole repo
    corpus — the only channel that can tell an enumerator where an ABSENT import
    lives, since the parent file by definition does not contain it. Omitting
    either argument never changes any other family's output, so every existing
    caller is unaffected.
    """
    families = set(_resolve_families(cls))
    names = (collect_span_names(span_lines)
             if "wrong_variable" in families else set())

    # Contact families are keyed per line so they slot into the SAME per-line cap,
    # appended after the seven measured families (never reordering them).
    contact_by_line = {}
    active_contact = [f for f in CONTACT_FAMILIES if f in families]
    # none_flip's AST arm rides the same per-line channel: it needs the parsed
    # span, and appending it here keeps it behind the text flip in the per-line
    # order (so a cap can only ever evict the NEW candidates).
    want_none_ast = "none_flip" in families
    want_attr = "wrong_attribute" in families
    # Expressive families ride the SAME per-line channel, appended after the
    # attribute family, so a per-line or per-function cap can only ever evict
    # them — never a candidate from a family that was already measured.
    active_expressive = [f for f in EXPRESSIVE_FAMILIES if f in families]
    if active_contact or want_none_ast or want_attr or active_expressive:
        facts = span_facts(span_lines)
        if facts is not None:
            for fam in active_contact:
                for off, old, new in CONTACT_GENERATORS[fam](facts):
                    if 0 <= off < len(span_lines) and span_lines[off] == old:
                        contact_by_line.setdefault(off, []).append(new)
            if want_none_ast:
                for off, old, new in _none_fstring_mutations(facts):
                    if 0 <= off < len(span_lines) and span_lines[off] == old:
                        contact_by_line.setdefault(off, []).append(new)
            if want_attr:
                # Evidence base: the whole file when the caller supplied it, always
                # unioned with the span's own usage (a span-only pool is what the
                # span already proves, and it keeps the family alive for callers
                # that pass no file context).
                span_names = collect_span_names(span_lines)
                pool = collect_attribute_pool(span_lines, span_names)
                for attr, ent in collect_attribute_pool(
                        file_lines or [], span_names).items():
                    cur = pool.setdefault(attr, {"recv": set(), "count": 0})
                    cur["recv"].update(ent["recv"])
                    cur["count"] = max(cur["count"], ent["count"])
                for off, old, new in _wrong_attribute(facts, pool, span_names):
                    if 0 <= off < len(span_lines) and span_lines[off] == old:
                        contact_by_line.setdefault(off, []).append(new)
            for fam in active_expressive:
                gen = EXPRESSIVE_GENERATORS[fam]
                if fam == "literal_substitution":
                    produced = gen(facts, collect_literal_pool(file_lines))
                elif fam == "clause_insertion":
                    produced = gen(facts, file_lines)
                else:
                    produced = gen(facts)
                for off, old, new in produced:
                    if 0 <= off < len(span_lines) and span_lines[off] == old:
                        contact_by_line.setdefault(off, []).append(new)

    by_line = []
    for off, line in enumerate(span_lines):
        if not line.strip():
            continue
        per_line = []
        per_line += _legacy_mutations(line, families)
        if "off_by_one" in families:
            per_line += _off_by_one_mutations(line)
        if "wrong_variable" in families:
            per_line += _wrong_variable_mutations(line, names)
        if "missing_not" in families:
            per_line += _missing_not_mutations(line)
        per_line += contact_by_line.get(off, [])
        capped = _cap(per_line, MAX_PER_LINE, line)
        if capped:
            by_line.append([(start_idx + off, line, new) for new in capped])

    # PER-FUNCTION CAP, ROUND-ROBIN ACROSS LINES.
    #
    # Measured gap (2026-08-29, gym task api/study_corner.py::get_course_gaps):
    # the previous `dedup[:MAX_PER_FUNCTION]` truncated in LINE ORDER, so on an
    # unrouted (cls=None) enumeration the first few lines of a long function spent
    # the entire 60-candidate budget and the gold `and` -> `or` flip — the ONLY
    # candidate on the 23rd line — was never in the menu at all. Line order is not
    # a relevance order, and a bug is no likelier to sit at the top of a function.
    #
    # Round-robin makes the budget fair: every line contributes its 1st candidate
    # before any line contributes its 2nd. WITHIN a line the existing order is
    # untouched, so the family precedence (measured seven, then contact, then
    # attribute) still decides which of a line's own candidates gets evicted.
    flat, rank = [], 0
    while len(flat) < MAX_PER_FUNCTION:
        progressed = False
        for cands in by_line:
            if rank >= len(cands):
                continue
            progressed = True
            flat.append(cands[rank])
            if len(flat) >= MAX_PER_FUNCTION:
                break
        if not progressed:
            break
        rank += 1

    seen, dedup = set(), []
    for li, old, new in flat:
        if (li, new) not in seen and new != old:
            seen.add((li, new))
            dedup.append((li, old, new))

    # COORDINATED EDIT-SETS — APPENDED AFTER THE PER-FUNCTION CAP, out of their
    # own reserved budget (edit_sets.MAX_EDIT_SETS). Deliberately NOT routed
    # through `by_line`/`_cap`: an edit-set's `old` is a multi-line BLOCK, not
    # `span_lines[off]`, so the per-line channel would silently rewrite its
    # anchor. Appending here is also the strongest possible no-regression
    # guarantee — the single-line list above is already final and byte-identical
    # to what it was before this family existed.
    if "coordinated_edit_set" in families:
        try:
            for es in edit_sets.enumerate_edit_sets(span_lines, start_idx,
                                                    file_lines,
                                                    providers=providers):
                if (es[0], es[2]) not in seen:
                    seen.add((es[0], es[2]))
                    dedup.append(es)
        except Exception:                # law 1: a broken family learns nothing
            pass
    return dedup
