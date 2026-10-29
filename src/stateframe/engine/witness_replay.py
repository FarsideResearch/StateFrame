"""WITNESS REPLAY — code decides boolean fixes; the model only breaks ties.

Design authority: sol_answers/a_wobble.md Q3.

    "Replay each candidate against the captured witness plus a tiny boundary set:
     true/false combinations, null, empty, zero, and missing values. If exactly one
     candidate matches expected behavior without regressions, let code select it
     directly; send only unresolved ties to the 1B model."

The measured problem this attacks: on vague-symptom tasks ("combines its conditions
incorrectly") the mutation menu carries 4+ similar boolean lines, the 1B's top-2
logit margin collapses to <0.5 nats, and the pick flips between server boots and
backends. That wobble is a MODEL problem only because we were asking the model a
question CODE can answer: which of these predicates actually behaves differently,
and in the direction the symptom describes.

Mechanism (all deterministic, no network, no execution of repo code)
-------------------------------------------------------------------
1. CONDITION EXTRACTION — parse the candidate's old and new line with `ast` (inside
   a dummy `... : pass` context when the line is a bare header) and pull the
   governing expression out of if/elif/while/return/assign/assert/bare-expr.
2. WITNESS DERIVATION — collect the condition's free variables (dotted attribute
   chains are ONE opaque variable, never decomposed) and take the cartesian product
   of the boundary value set {None, 0, 1, -1, "", "x", [], [1], True, False},
   shrinking the per-variable set so the product stays under MAX_EVALS (200).
3. SAFE EVALUATION — a whitelist interpreter (Compare / BoolOp / UnaryOp-not /
   Name / Constant / opaque Attribute, plus explicit `len(x)` and `x.strip()`).
   ANY node outside the whitelist makes the candidate UNSUPPORTED, never an error.
   A TypeError from a real comparison (`None < 1`) is a legitimate row outcome and
   is recorded as an ERR row, not a failure.
4. TRUTH-TABLE DELTA — the witness rows where the new condition disagrees with the
   old one. An EMPTY delta means the candidate changes nothing (`x < y` -> `y > x`
   is a mirror, not a fix) and is eliminated unconditionally.
5. SYMPTOM MATCH — the task's own symptom words name a bug class; each class has an
   expected AST-shape (and for comparisons, an expected DIRECTION reversal). A
   candidate whose shape contradicts the class is eliminated.

FAIL-OPEN IS THE LAW. Anything we cannot evaluate SURVIVES. This filter may only
ever remove candidates it can positively prove are wrong; the cost of a false
elimination is an unfixable task, the cost of a false survival is one model call.

Deliberately NOT done: choosing between `a < b -> a > b` and `a < b -> a >= b`.
Both reverse the direction and, from the line alone, both are consistent with
"threshold applied in the wrong direction" (the harvest injector produces BOTH
shapes: `>=` -> `<` is a complement, `>` -> `<` is a strict mirror). Code narrows
the menu to those two and hands the genuine ambiguity to the model. Manufacturing a
unique survivor there would be a guess wearing a proof's clothes.

Public contract:
    filter_candidates(task_text, span_lines, muts) -> (survivors, evidence)
      muts      : [(line_idx, old_line, new_line)]
      survivors : the subset of `muts` (same tuples, same order) that survived
      evidence  : one dict per candidate, index-aligned with `muts`
"""
import ast
import itertools
import re
import textwrap

# --------------------------------------------------------------------- budget
MAX_EVALS = 200          # witness rows evaluated per candidate (hard ceiling)
MAX_FREE_VARS = 8        # beyond this even a 2-value set blows the ceiling

# Boundary witnesses per a_wobble Q3: "true/false combinations, null, empty, zero,
# and missing values". Ordered by disambiguating power so a shrunk set keeps the
# rows that actually separate predicates.
WITNESS_VALUES = (None, 0, 1, "", [], True, False, -1, "x", [1])


# ------------------------------------------------------------------ exceptions
class Unsupported(Exception):
    """The candidate is outside the evaluable subset. It SURVIVES."""


# ------------------------------------------------- 1. condition extraction
def extract_condition(line):
    """The governing expression of a single source line, or None.

    Handles bare block headers (`if x and y:`) by parsing them with a `pass` body.
    Returns an ast expression node, plus a flag saying whether the line was a
    conditional TEST (as opposed to a returned/assigned value).
    """
    if not isinstance(line, str):
        return None, False
    stripped = textwrap.dedent(line).strip()
    if not stripped or stripped.startswith("#"):
        return None, False
    tree = None
    for src in (stripped, stripped + " pass", stripped + "\n    pass"):
        try:
            tree = ast.parse(src)
            break
        except SyntaxError:
            continue
    if tree is None or not tree.body:
        return None, False
    node = tree.body[0]
    if isinstance(node, (ast.If, ast.While)):
        return node.test, True
    if isinstance(node, ast.Assert):
        return node.test, True
    if isinstance(node, ast.Return):
        return node.value, False
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return node.value, False
    if isinstance(node, ast.Expr):
        return node.value, False
    return None, False


def _is_boolean_shaped(node):
    """Only boolean/comparison/None-check/not candidates are in scope (the task's
    step 1). An arithmetic or string expression is left entirely alone."""
    if node is None:
        return False
    for n in ast.walk(node):
        if isinstance(n, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return True
    return False


# --------------------------------------------------- 2. witness derivation
def free_variables(*nodes):
    """Free variable KEYS of the given expressions.

    A dotted attribute chain (`user.profile.age`) is ONE opaque variable keyed by
    its dotted text — we never model attribute semantics, we just bind the whole
    chain to a witness value. A non-trivial attribute base (`f(x).y`) is
    unsupported.
    """
    keys = set()
    for node in nodes:
        if node is None:
            continue
        _collect_vars(node, keys)
    return sorted(keys)


def _dotted(node):
    """`a.b.c` -> "a.b.c"; anything else -> None."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _collect_vars(node, keys):
    if isinstance(node, ast.Name):
        keys.add(node.id)
        return
    if isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        if dotted is None:
            raise Unsupported("attribute chain with a non-name base")
        keys.add(dotted)
        return
    if isinstance(node, ast.Call):
        # `len(x)` and `x.strip()` are the only calls we model; their arguments
        # still contribute variables.
        _collect_call_vars(node, keys)
        return
    for child in ast.iter_child_nodes(node):
        _collect_vars(child, keys)


# PURE BUILTINS (rung-4000 autopsy, ranked fix #8). MEASURED: the gold line of
# `math_answer_verifier::_build_equivalent_forms` is
#     if abs(numer / denom - val) > 1e-9:
# and witness replay reported `unsupported: call outside the whitelist:
# Name(id='abs')`, which cost the candidate its truth table and let the
# same-line boundary tie-break eliminate it. Every name below is a pure,
# side-effect-free, deterministic builtin over the witness value set — replaying
# one cannot touch the filesystem, the network or any object we did not bind
# ourselves — so admitting them is a strict widening of what CODE can decide,
# never a widening of what code may DO. Anything they raise on a witness row
# (`abs(None)`, `int("x")`) is caught upstream by `_row_key` and recorded as an
# honest ERR outcome, exactly like `None < 1` already is.
#
# `len` keeps its historical one-argument form (it predates this table and is
# listed here so the whitelist has ONE source of truth); the rest take one
# positional argument, no keywords. `min`/`max` additionally accept the
# two-argument form, which is how they appear in real boundary code.
PURE_BUILTINS = {
    "abs": 1, "len": 1, "round": 1, "int": 1, "float": 1, "str": 1,
    "min": 2, "max": 2,
}


def _pure_builtin_call(fn, node):
    """(name, nargs) when `node` is a whitelisted pure-builtin call, else None."""
    if not isinstance(fn, ast.Name) or node.keywords:
        return None
    maxargs = PURE_BUILTINS.get(fn.id)
    if maxargs is None:
        return None
    if not node.args or len(node.args) > maxargs:
        return None
    return fn.id, len(node.args)


def _collect_call_vars(node, keys):
    fn = node.func
    if _pure_builtin_call(fn, node) is not None:
        for arg in node.args:
            _collect_vars(arg, keys)
        return
    if isinstance(fn, ast.Attribute) and fn.attr == "strip" and not node.args \
            and not node.keywords:
        _collect_vars(fn.value, keys)
        return
    raise Unsupported(f"call outside the whitelist: {ast.dump(fn)[:40]}")


def witness_rows(var_keys):
    """Bounded cartesian product of boundary values over the free variables.

    Shrinks the per-variable value set (keeping the highest-signal values first)
    until the product fits MAX_EVALS. Zero variables -> one empty binding, which
    is correct: a constant condition has exactly one behavior.
    """
    k = len(var_keys)
    if k == 0:
        return [{}]
    if k > MAX_FREE_VARS:
        raise Unsupported(f"{k} free variables exceeds the witness budget")
    m = len(WITNESS_VALUES)
    while m > 1 and m ** k > MAX_EVALS:
        m -= 1
    if m ** k > MAX_EVALS:
        raise Unsupported(f"{k} free variables exceeds the witness budget")
    values = WITNESS_VALUES[:m]
    return [dict(zip(var_keys, combo))
            for combo in itertools.product(values, repeat=k)]


# ------------------------------------------------------ 3. safe evaluation
_CMP_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def safe_eval(node, env):
    """Whitelist interpreter. Raises Unsupported for any node outside it.

    Runtime errors from legitimate operations (`None < 1`) are NOT swallowed here;
    the caller turns them into an ERR witness row.
    """
    if node is None:
        raise Unsupported("empty expression")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise Unsupported(f"unbound name {node.id}")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        dotted = _dotted(node)
        if dotted is None or dotted not in env:
            raise Unsupported("unbound attribute chain")
        return env[dotted]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not safe_eval(node.operand, env)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return -node.operand.value
        raise Unsupported("unary operator outside the whitelist")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            out = True
            for v in node.values:
                out = safe_eval(v, env)
                if not out:
                    return out
            return out
        if isinstance(node.op, ast.Or):
            out = False
            for v in node.values:
                out = safe_eval(v, env)
                if out:
                    return out
            return out
        raise Unsupported("boolean operator outside the whitelist")
    if isinstance(node, ast.Compare):
        left = safe_eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise Unsupported("comparison operator outside the whitelist")
            right = safe_eval(comparator, env)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BinOp):
        # ARITHMETIC INSIDE A COMPARISON (rung-4000 fix #8, second half). `abs()`
        # alone buys nothing on the measured gold line: its argument is
        # `numer / denom - val`, a BinOp, so the candidate stayed unsupported.
        # Only the pure numeric operators are admitted, each of them total over
        # the witness value set except for the errors `_row_key` already models
        # (ZeroDivisionError, TypeError on `None - 1`) — those become ERR rows,
        # which is a REAL observable difference between two candidates, not a
        # crash. Bit/shift/matmul are deliberately absent: no repair family in
        # this harness emits them, and an operator nothing produces is a reader
        # with no writer.
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise Unsupported("binary operator outside the whitelist")
        return op(safe_eval(node.left, env), safe_eval(node.right, env))
    if isinstance(node, ast.Call):
        return _safe_call(node, env)
    raise Unsupported(f"node outside the whitelist: {type(node).__name__}")


def _safe_call(node, env):
    fn = node.func
    pure = _pure_builtin_call(fn, node)
    if pure is not None:
        name, _n = pure
        args = [safe_eval(a, env) for a in node.args]
        # __builtins__ is never consulted: the callable comes from an explicit
        # table, so no name lookup the model wrote can reach anything else.
        impl = {"abs": abs, "len": len, "round": round, "int": int,
                "float": float, "str": str, "min": min, "max": max}[name]
        return impl(*args)
    if isinstance(fn, ast.Attribute) and fn.attr == "strip" and not node.args \
            and not node.keywords:
        target = safe_eval(fn.value, env)
        if not isinstance(target, str):
            raise TypeError("strip() on a non-string witness")
        return target.strip()
    raise Unsupported("call outside the whitelist")


def _row_key(node, env):
    """The observable outcome of one witness row: a hashable, type-aware key so
    `True` and `1` are never conflated (0 == False would otherwise hide a delta)."""
    try:
        value = safe_eval(node, env)
    except Unsupported:
        raise
    except Exception as exc:                       # a real runtime outcome
        return ("ERR", type(exc).__name__)
    return ("OK", type(value).__name__, repr(value))


def truth_table(node, rows):
    return [_row_key(node, env) for env in rows]


# ------------------------------------------------------- 4. delta / shape
_CMP_TOKENS = ("Lt()", "LtE()", "Gt()", "GtE()", "Eq()", "NotEq()")
_IS_TOKENS = ("Is()", "IsNot()")
_BOOL_TOKENS = ("And()", "Or()")

# ordering direction of a comparison operator; None = not an ordering operator
_DIRECTION = {ast.Lt: "lt", ast.LtE: "lt", ast.Gt: "gt", ast.GtE: "gt"}


def _norm(dump, tokens, placeholder):
    for t in tokens:
        dump = dump.replace(t, placeholder)
    return dump


def classify_shape(old_node, new_node):
    """What KIND of edit turned old into new. One of:
    cmp_op_change / bool_op_change / none_change / not_change / identical / other.
    """
    d_old, d_new = ast.dump(old_node), ast.dump(new_node)
    if d_old == d_new:
        return "identical"
    # negation added or removed around an otherwise identical expression
    for a, b in ((old_node, new_node), (new_node, old_node)):
        if isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.Not) \
                and ast.dump(b.operand) == ast.dump(a):
            return "not_change"
    if _norm(d_old, _IS_TOKENS, "ISOP") == _norm(d_new, _IS_TOKENS, "ISOP") \
            and _has_none_identity(old_node):
        return "none_change"
    if _norm(d_old, _CMP_TOKENS, "CMPOP") == _norm(d_new, _CMP_TOKENS, "CMPOP"):
        return "cmp_op_change"
    if _norm(d_old, _BOOL_TOKENS, "BOOLOP") == _norm(d_new, _BOOL_TOKENS, "BOOLOP"):
        return "bool_op_change"
    return "other"


def _has_none_identity(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Compare) and any(
                isinstance(o, (ast.Is, ast.IsNot)) for o in n.ops):
            if any(isinstance(c, ast.Constant) and c.value is None
                   for c in n.comparators):
                return True
    return False


def _cmp_ops(node):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            out.extend(type(o) for o in n.ops)
    return out


def reverses_direction(old_node, new_node):
    """True when every CHANGED ordering operator flips its direction (< family <->
    > family) and at least one did. `< -> <=` is a boundary tweak, not a direction
    reversal, and is therefore NOT a "wrong direction" fix."""
    a, b = _cmp_ops(old_node), _cmp_ops(new_node)
    if len(a) != len(b):
        return False
    changed = 0
    for oa, ob in zip(a, b):
        if oa is ob:
            continue
        changed += 1
        da, db = _DIRECTION.get(oa), _DIRECTION.get(ob)
        if da is None or db is None or da == db:
            return False
    return changed > 0


# --------------------------------------------------------- 5. symptom class
# Symptom words -> bug class. Keys mirror probes/real_repo_baseline.py:SYMPTOMS and
# the mutation families, so production ticket prose and the synthetic templates land
# in the same buckets. Checked in order; the FIRST class whose pattern hits wins.
SYMPTOM_PATTERNS = (
    ("missing_not", r"\bmissing (?:a )?not\b|\bnegat|\binvert(?:ed|s|ing)?\b"
                    r"|\bshould be negated\b"),
    ("none_flip", r"\bnone\b|\bnull\b|\bmissing[ /]|\bmissing value|\babsent\b"),
    ("cmp_flip", r"\bthreshold\b|\bboundary\b|\bwrong direction\b|\bcomparison\b"
                 r"|\bbackwards comparison\b|\bgreater than\b|\bless than\b"),
    ("bool_flip", r"\bcombines? (?:its )?conditions\b|\band/or\b|\bconditions?"
                  r" (?:are )?(?:combined|incorrect)|\bboolean logic\b"),
)


def infer_symptom_class(task_text):
    """The bug class the SYMPTOM describes, or None when the prose does not say.

    None is a first-class answer: with no inferable class every candidate with a
    non-empty truth-table delta survives.
    """
    text = (task_text or "").lower()
    for cls, pattern in SYMPTOM_PATTERNS:
        if re.search(pattern, text):
            return cls
    return None


def _matches_class(cls, shape, old_node, new_node):
    """Is this candidate's edit consistent with the symptom class? (reason, ok)."""
    if cls == "cmp_flip":
        if shape != "cmp_op_change":
            return False, f"class cmp_flip expects a comparison change, got {shape}"
        if not reverses_direction(old_node, new_node):
            return False, "comparison changed but the direction did not reverse"
        return True, "mirrors the comparison direction"
    if cls == "none_flip":
        if shape != "none_change":
            return False, f"class none_flip expects an is/is-not None flip, got {shape}"
        return True, "flips the None identity check"
    if cls == "bool_flip":
        if shape != "bool_op_change":
            return False, f"class bool_flip expects an and/or swap, got {shape}"
        return True, "swaps the boolean connective"
    if cls == "missing_not":
        if shape != "not_change":
            return False, f"class missing_not expects a not insertion/removal, got {shape}"
        return True, "inserts or removes the negation"
    return True, "no class constraint"


# ------------------------------------------------- 6. contact-family witnesses
# The five contact families (mutations.CONTACT_FAMILIES) are STRUCTURAL edits, so
# the boolean truth-table machinery above cannot judge them — a guard-insertion
# candidate is not even a single line. Each gets its own witness: a predicate that
# re-derives the family's premise from the SPAN ITSELF and kills the candidate when
# the premise is false. Crucially the witness never trusts the generator: it
# re-parses and re-checks, so a candidate that reached us from a model, a stale
# menu or a buggy generator is judged on the same evidence.
#
# The fail-open law still holds: a witness that cannot decide returns (False, "")
# meaning "does not apply", and the candidate carries on to the existing logic.

def _span_facts(span_lines):
    try:
        from mutations import span_facts
    except Exception:                       # mutations unavailable -> no opinion
        return None
    try:
        return span_facts(span_lines)
    except Exception:
        return None


_KEYWORDS = frozenset((
    "and", "or", "not", "in", "is", "if", "else", "elif", "while", "for",
    "return", "None", "True", "False", "await", "async", "def", "lambda",
    "yield", "assert", "raise", "pass", "break", "continue"))


def _parses(text):
    """Does this source parse? Bare block headers (`if x and y:`) are given the
    same `pass` body treatment extract_condition uses, so a legitimate header is
    never mistaken for a syntax error."""
    stripped = textwrap.dedent(text).strip()
    if not stripped:
        return False
    for src in (stripped, stripped + " pass", stripped + "\n    pass"):
        try:
            ast.parse(src)
            return True
        except SyntaxError:
            continue
    return False


def _enclosing_fn(facts, off):
    fn = facts.enclosing.get(off)
    if fn is not None:
        return fn
    best = None
    for node in ast.walk(facts.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo = node.lineno - 1
            hi = getattr(node, "end_lineno", node.lineno) - 1
            if lo <= off <= hi and (best is None or lo > best.lineno - 1):
                best = node
    return best


def _guard_witness(facts, off, old_line, new_line):
    """Guard insertion: the block must parse, the guarded symbol must be BOUND
    STRICTLY ABOVE the insertion point, the boundary must be a real statement start
    at the target line's indent, the guard must not duplicate one already present,
    and the original line must survive intact as the tail."""
    parts = new_line.split("\n")
    if len(parts) < 3 or parts[-1] != old_line:
        return True, False, "guard insertion did not preserve the guarded statement"
    head = parts[0]
    m = re.match(r"^(\s*)if\s+(.+?):\s*$", head)
    if not m:
        return True, False, "guard insertion head is not a well-formed `if`"
    indent, cond = m.groups()
    target_indent = old_line[:len(old_line) - len(old_line.lstrip())]
    if indent != target_indent:
        return True, False, ("guard indent %r does not match the boundary %r"
                             % (indent, target_indent))
    if not _parses("\n".join(parts)):
        return True, False, "guard block does not parse"
    if facts is None:
        return True, True, "guard block parses (no span facts to check scope)"
    if off not in facts.enclosing:
        return True, False, "insertion point is not a statement boundary"
    try:
        cond_node = ast.parse(cond, mode="eval").body
    except SyntaxError:
        return True, False, "guard condition does not parse"
    for n in ast.walk(cond_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if not facts.defined_before(n.id, off):
                return True, False, (
                    "guard references %r, which is not bound above the boundary"
                    % n.id)
    norm = re.sub(r"\s+", " ", head.strip())
    for i, line in enumerate(facts.span_lines):
        if i != off and re.sub(r"\s+", " ", line.strip()) == norm:
            return True, False, "an equivalent guard already exists in the span"
    fn = _enclosing_fn(facts, off)
    body = "\n".join(parts[1:-1])
    if fn is not None and re.search(r"\breturn\b(?!\s*(#|$))", body):
        # a value-returning guard inside a generator would be a semantic change
        if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)):
            return True, False, "value return inserted into a generator"
    return True, True, "guard is scoped, non-duplicate and return-compatible"


_AWAIT_RE = re.compile(r"(?<![\w.])await\s+")


def _await_witness(facts, off, old_line, new_line):
    """missing_await: the caller must be `async def` and the callee must have
    span-local awaitable evidence. NEVER fires in a sync function."""
    if _AWAIT_RE.sub("", new_line, count=1) != old_line:
        return False, False, ""
    if facts is None:
        return True, False, "cannot verify the async boundary for an await insertion"
    fn = _enclosing_fn(facts, off)
    if not isinstance(fn, ast.AsyncFunctionDef):
        return True, False, "await inserted in a function that is not `async def`"
    if not _parses(new_line):
        return True, False, "await insertion does not parse"
    try:
        node = ast.parse(textwrap.dedent(new_line)).body[0]
    except (SyntaxError, IndexError):
        return True, False, "await insertion does not parse"
    from mutations import _callee_name
    names = {_callee_name(n.value) for n in ast.walk(node)
             if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)}
    names.discard(None)
    if not names:
        return True, False, "await does not wrap a call"
    evidence = names & (facts.async_defs | facts.awaited)
    if not evidence:
        return True, False, ("no span-local awaitable evidence for %s"
                             % sorted(names))
    return True, True, "callee %s is awaitable and the caller is async" % sorted(evidence)


def _boolop_witness(old_line, new_line):
    """boolean_operator: an and<->or swap must FLIP at least one row of the truth
    table over its operands. `a and a` -> `a or a` changes nothing and dies here."""
    old_node, _ = extract_condition(old_line)
    new_node, _ = extract_condition(new_line)
    if old_node is None or new_node is None:
        return False, False, ""
    if classify_shape(old_node, new_node) != "bool_op_change":
        return False, False, ""
    try:
        rows = witness_rows(free_variables(old_node, new_node))
        t_old, t_new = truth_table(old_node, rows), truth_table(new_node, rows)
    except Unsupported:
        return False, False, ""
    flipped = sum(1 for a, b in zip(t_old, t_new) if a != b)
    if not flipped:
        return True, False, ("and/or swap changes nothing over %d truth-table rows"
                             % len(rows))
    return True, True, "and/or swap flips %d/%d truth-table rows" % (flipped, len(rows))


def _parse_line_stmt(line):
    """The single statement this source line contains, or None. Bare block headers
    get the same `pass` body treatment `extract_condition` uses."""
    if not isinstance(line, str):
        return None
    stripped = textwrap.dedent(line).strip()
    if not stripped or stripped.startswith("#"):
        return None
    for src in (stripped, stripped + " pass", stripped + "\n    pass"):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        return tree.body[0] if tree.body else None
    return None


def _non_load_identifiers(line):
    """Identifiers on this line that are NOT free `ast.Name` LOADS.

    `x.attr` binds nothing called `attr` — it is a member of whatever `x` is, and
    the span's symbol table has no business being consulted about it. Same for
    keyword-argument names (`f(attr=1)`) and for the STORE targets of the line
    itself: a name being assigned ON the mutated line is not a use that some
    earlier binding has to dominate.

    Returns None when the line does not parse — the caller then has no positive
    evidence and must not eliminate on this basis (the fail-open law).
    """
    node = _parse_line_stmt(line)
    if node is None:
        return None
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.arg,)):
            out.add(n.arg)
    return out


def _one_node_diff(facts, off, old_line, new_line):
    """(old_names, new_names) when exactly one free-NAME token differs, else None.

    Attribute identifiers, keyword-argument names and the line's own store targets
    are explicitly NOT symbol substitutions (see `_non_load_identifiers`): claiming
    them here made the dominance witness eliminate every candidate of an attribute
    menu (`msg.message` -> `msg.content`), gold included.
    """
    if facts is None:
        return None
    ot = re.findall(r"[A-Za-z_]\w*", old_line)
    nt = re.findall(r"[A-Za-z_]\w*", new_line)
    if len(ot) != len(nt):
        return None
    diff = [(a, b) for a, b in zip(ot, nt) if a != b]
    if len(diff) != 1:
        return None
    # A keyword on either side is an OPERATOR edit (and/or, not, is), not a symbol
    # substitution — that belongs to a different family's witness.
    if diff[0][0] in _KEYWORDS or diff[0][1] in _KEYWORDS:
        return None
    old_non_load = _non_load_identifiers(old_line)
    new_non_load = _non_load_identifiers(new_line)
    if old_non_load is None or new_non_load is None:
        return None                         # cannot re-derive -> no opinion
    if diff[0][0] in old_non_load or diff[0][1] in new_non_load:
        return None                         # attribute / kwarg / def name / param
    # A name being STORED on the mutated line is still a real dominance question
    # for the NEW symbol (`y = g(y)` must die), so store targets deliberately stay
    # OUT of the exclusion set above.
    return diff[0]


def _wrong_variable_used_witness(facts, off, old_line, new_line):
    """The substituted symbol must be BOUND STRICTLY ABOVE the use (dominance) and
    the edit must be a one-node diff that still parses."""
    diff = _one_node_diff(facts, off, old_line, new_line)
    if diff is None:
        return False, False, ""
    old_name, new_name = diff
    if not _parses(new_line) and not _parses(new_line + "\n    pass"):
        return True, False, "one-name substitution does not parse"
    if new_name in ("None", "True", "False") or old_name in ("None", "True", "False"):
        return False, False, ""
    if new_name not in facts.sym_def_off:
        return True, False, "%r is not bound anywhere in the span" % new_name
    if not facts.defined_before(new_name, off):
        return True, False, ("%r is not defined above its use (no dominance)"
                             % new_name)
    return True, True, "%r -> %r: bound above the use, one-node diff" % (
        old_name, new_name)


def _kwarg_witness(facts, off, old_line, new_line):
    """missing_kwarg: the new call must bind CLEANLY against the span-resolved
    signature — the parameter must exist, must not already be supplied, and the
    source symbol must be in scope."""
    m = re.search(r"[,(]\s*([A-Za-z_]\w*)=([A-Za-z_][\w.]*)\s*\)", new_line)
    if m is None or facts is None:
        return False, False, ""
    ins = new_line[:m.start(1)] + new_line[m.end(2):]
    if re.sub(r"[,\s]", "", ins) != re.sub(r"[,\s]", "", old_line):
        return False, False, ""
    param, source = m.group(1), m.group(2)
    if not _parses(new_line) and not _parses(new_line.strip()):
        return True, False, "call with the inserted keyword does not parse"
    try:
        node = ast.parse(textwrap.dedent(new_line).strip()).body[0]
    except (SyntaxError, IndexError):
        return True, False, "call with the inserted keyword does not parse"
    from mutations import _callee_name, _signature_params
    call = next((n for n in ast.walk(node)
                 if isinstance(n, ast.Call)
                 and any(k.arg == param for k in n.keywords)), None)
    if call is None:
        return False, False, ""
    fn = facts.defs.get(_callee_name(call))
    if fn is None:
        return True, False, "callee signature is not resolvable from the span"
    _required, allowed = _signature_params(fn)
    if param not in allowed:
        return True, False, "%r is not a parameter of the resolved signature" % param
    positional = set(allowed[:len(call.args)])
    if param in positional:
        return True, False, "%r is already supplied positionally (duplicate)" % param
    if sum(1 for k in call.keywords if k.arg == param) > 1:
        return True, False, "%r supplied twice" % param
    root = source.split(".")[0]
    if not facts.defined_before(root, off) and root not in facts.sym_def_off:
        return True, False, "source symbol %r is not in scope" % root
    return True, True, "%s=%s binds cleanly against the resolved signature" % (
        param, source)


def family_witness(span_lines, mut):
    """Structural witness for the contact families.

    Returns (applies, ok, reason). `applies=False` means this candidate is not a
    contact-family edit (or we could not tell) and the caller should fall through
    to the boolean machinery unchanged. Never raises.
    """
    try:
        off, old_line, new_line = mut
    except (TypeError, ValueError):
        return False, False, ""
    if not isinstance(old_line, str) or not isinstance(new_line, str):
        return False, False, ""
    span_lines = list(span_lines or [])
    # `off` is an ABSOLUTE file index; the span facts are span-relative, and the
    # span's start index is not part of the contract. Recover it by matching the
    # old line. When the line is ambiguous we take the LAST match: dominance is
    # easiest to satisfy deepest in the span, so an ambiguous span can never
    # produce a FALSE ELIMINATION (the fail-open law of this module).
    base = None
    for i, line in enumerate(span_lines):
        if line == old_line:
            base = i
    if base is None:
        return False, False, ""
    facts = _span_facts(span_lines)
    try:
        if "\n" in new_line:
            return _guard_witness(facts, base, old_line, new_line)
        # NOTE: and/or swaps are deliberately NOT claimed here. The boolean
        # machinery below already judges them (an empty truth-table delta kills a
        # tautological swap, and the symptom class still applies), and claiming
        # them would overwrite the `shape` field that pathway's callers read.
        # A witness that duplicates an existing judge only creates drift.
        for probe in (_await_witness, _wrong_variable_used_witness, _kwarg_witness):
            applies, ok, why = probe(facts, base, old_line, new_line)
            if applies:
                return applies, ok, why
        return False, False, ""
    except Unsupported:
        return False, False, ""
    except Exception:                       # a witness bug must never kill a fix
        return False, False, ""


# ------------------------------------------------------------- entry point
def editset_witness(span_lines, mut):
    """Judge a coordinated `EditSet` by RUNNING the composed function.

    (survived, reason). The set is applied atomically, the composed function is
    defined in a fresh namespace, and its behaviour is compared against the
    original over the same boundary-witness values every other judge uses. A set
    that changes NOTHING over every witness row is a mirror, not a fix, and is
    killed exactly as a single-line mirror is.

    Fail-OPEN on anything unevaluable (Unsupported survives, law 1): most real
    coordinated fixes touch ORM objects and DB sessions that no local witness can
    stand up, and a false kill is worse than no signal. The set-level structural
    checks live in `spec_exec._check_set_consistency`; this is the behavioural
    half, and it only speaks when it can actually run the function.
    """
    import edit_sets as _es
    composed = _es.apply_to_span(span_lines, mut)
    if composed is None:
        return False, "the edit-set does not apply atomically to this span"
    old_fn = _define_fn(span_lines)
    new_fn = _define_fn(composed)
    if old_fn is None or new_fn is None:
        return True, ("unsupported: the composed function cannot be defined "
                      "locally (structural checks still apply)")
    old_name, old_obj, old_params = old_fn
    new_name, new_obj, new_params = new_fn
    if old_name != new_name or old_params != new_params:
        return True, "unsupported: the set changed the signature"
    if not old_params or len(old_params) > MAX_FREE_VARS:
        return True, "unsupported: no witnessable parameter set"
    rows = witness_rows(tuple(old_params))
    seen_ok, flipped = 0, 0
    for row in rows[:MAX_EVALS]:
        args = [row[p] for p in old_params]
        a = _call_capture(old_obj, args)
        b = _call_capture(new_obj, args)
        if a is _UNRUNNABLE or b is _UNRUNNABLE:
            continue
        seen_ok += 1
        if a != b:
            flipped += 1
    if seen_ok == 0:
        return True, ("unsupported: the composed function never ran cleanly on "
                      "any witness row")
    if flipped == 0:
        return False, ("no behavior change over %d executed witness rows "
                       "(a mirror, not a fix)" % seen_ok)
    return True, ("the composed set changes behaviour on %d/%d executed witness "
                  "rows" % (flipped, seen_ok))


class _Unrunnable:
    def __repr__(self):                                   # pragma: no cover
        return "<unrunnable>"


_UNRUNNABLE = _Unrunnable()


def _define_fn(lines):
    """(name, callable, params) for the span's first function, or None.

    Executed in a namespace whose missing globals resolve to a permissive stub,
    so a function that merely NAMES an unavailable import can still be run. The
    stub is inert: it records nothing and reaches nothing outside the namespace.
    """
    src = textwrap.dedent("\n".join(lines))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node
            break
    if fn is None or isinstance(fn, ast.AsyncFunctionDef):
        return None
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg \
            or fn.args.kwonlyargs or fn.args.defaults:
        return None
    params = [a.arg for a in fn.args.args]
    ns = _StubNamespace()
    try:
        exec(compile(tree, "<editset-witness>", "exec"), ns)   # noqa: S102
    except Exception:
        return None
    obj = ns.get(fn.name)
    return (fn.name, obj, params) if callable(obj) else None


class _StubNamespace(dict):
    """Globals whose unknown names resolve to an inert callable/mapping stub."""

    def __missing__(self, key):
        if key.startswith("__"):
            raise KeyError(key)
        return _Stub()


class _Stub:
    def __call__(self, *a, **k):
        return _Stub()

    def __getattr__(self, name):
        return _Stub()

    def __getitem__(self, key):
        return _Stub()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, _Stub)

    def __hash__(self):
        return hash("<stub>")

    def __repr__(self):                                   # pragma: no cover
        return "<stub>"


def _call_capture(fn, args):
    """The call's result, or `_UNRUNNABLE`. Never raises."""
    try:
        out = fn(*args)
    except Exception:
        return _UNRUNNABLE
    try:
        hash(repr(out))
    except Exception:
        return _UNRUNNABLE
    return repr(out)


def _intent_oracle_enabled(cfg):
    """The knob, read defensively. Absent cfg / absent field -> OFF."""
    facts = getattr(cfg, "facts", None)
    return bool(getattr(facts, "intent_oracle", False))


def _apply_intent_oracle(span_lines, muts, survivors, evidence):
    """ADDITIVE HOOK (knob: facts.intent_oracle). Consults intent_oracle to kill
    survivors that contradict the function's own docstring contract.

    Contract with this module's fail-open law: any exception, any missing module,
    zero expectations, or fewer than two survivors leaves BOTH `survivors` and
    `evidence` exactly as they were. The oracle can only ever remove candidates it
    positively discriminated against a passing sibling, so it can never empty the
    list. Evidence rows gain `intent_oracle` / `intent_line` fields. For a row the
    oracle KILLED, and only for such a row, `survived` flips to False and `reason`
    is EXTENDED (never replaced — the witness-replay reason stays readable, so the
    certificate always shows both judges).
    """
    try:
        import intent_oracle
        fn_source = "\n".join(span_lines or [])
        res = intent_oracle.discriminate(survivors, fn_source, span_lines)
        lines = intent_oracle.render_evidence(res)
    except Exception:
        return survivors, evidence
    by_line = {}
    for cand, line in zip(res.get("candidates") or [], lines or []):
        by_line[id(cand)] = line
    killed_ids = {id(c) for c, _e, _w in (res.get("killed") or [])}
    out_survivors, out_evidence = [], []
    surv_ids = {id(s) for s in survivors}
    for ev, mut in zip(evidence, muts or []):
        if id(mut) in surv_ids and id(mut) in by_line:
            ev = dict(ev)
            ev["intent_line"] = by_line[id(mut)]
            if id(mut) in killed_ids:
                ev["intent_oracle"] = False
                ev["survived"] = False
                verdict = by_line[id(mut)] or "contradicts the stated intent"
                ev["reason"] = ("%s | intent oracle: %s" % (ev.get("reason", ""),
                                                            verdict)).strip(" |")
            else:
                ev["intent_oracle"] = True
        out_evidence.append(ev)
    for mut in survivors:
        if id(mut) not in killed_ids:
            out_survivors.append(mut)
    if not out_survivors:                 # unreachable; the law wins regardless
        return survivors, evidence
    return out_survivors, out_evidence


# ------------------------------------------------ 6b. mirror PRE-FILTER
# Measured harm this exists to stop: the harness BREAKING already-correct lines.
# `api/transcripts.py::build_course_row` had `course.course_level or "Standard"`
# rewritten to `and "Standard"` on three consecutive runs; `code_submissions`
# inverted an already-correct `passed = (score >= 70)`. In both the candidate's
# BEFORE state already reads in the direction the symptom asks for, so the edit is
# a mirror, not a fix — and it must never reach the menu, because once it is on the
# menu a 1B model picking it is a coin flip we lose half the time.
#
# Both rules are POSITIVE-EVIDENCE only and fail open: anything unparseable, any
# shape we cannot re-derive, any exception -> not a mirror, candidate carries on.

# Names whose truth conventionally INCREASES with the measured quantity. A
# comparison bound to one of these with `>`/`>=` is the conventional direction.
_POSITIVE_PREDICATE = re.compile(
    r"^(?:is|has|can|should|did)_\w+$|^(?:passed|passes|valid|ok|okay|allowed"
    r"|eligible|success|succeeded|approved|complete|completed|qualified)$")

_STRICT_OPS = (ast.Lt, ast.Gt)
_GREATER_OPS = (ast.Gt, ast.GtE)
_LESSER_OPS = (ast.Lt, ast.LtE)


def _default_idiom_mirror(old_line, new_line):
    """`X or <default>` swapped to `and`: the `or` is a default-value idiom in a
    VALUE position, and `and` cannot be the fix for anything."""
    old_node, old_is_test = extract_condition(old_line)
    new_node, _ = extract_condition(new_line)
    if old_node is None or new_node is None or old_is_test:
        return None
    if classify_shape(old_node, new_node) != "bool_op_change":
        return None
    for n in ast.walk(old_node):
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or) \
                and isinstance(n.values[-1], ast.Constant) \
                and n.values[-1].value not in (None, False):
            return ("`or <default>` is a default-value idiom already reading the "
                    "way the symptom asks; swapping it to `and` is a mirror, not "
                    "a fix")
    return None


def _positive_predicate_mirror(old_line, new_line):
    """`passed = score >= 70` flipped to `<`: the assigned name is a positive
    predicate and the BEFORE comparison already runs in the conventional
    direction, so reversing it breaks a correct line."""
    stmt = _parse_line_stmt(old_line)
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return None
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    if not _POSITIVE_PREDICATE.match(targets[0].id):
        return None
    old_node, _ = extract_condition(old_line)
    new_node, _ = extract_condition(new_line)
    if old_node is None or new_node is None:
        return None
    if classify_shape(old_node, new_node) != "cmp_op_change":
        return None
    a, b = _cmp_ops(old_node), _cmp_ops(new_node)
    if not a or len(a) != len(b):
        return None
    if not all(o in _GREATER_OPS for o in a):
        return None                         # before is NOT the conventional way
    if not any(o in _LESSER_OPS for o in b):
        return None
    return ("%r is a positive predicate and the comparison already runs in the "
            "conventional direction; reversing it is a mirror, not a fix"
            % targets[0].id)


def is_mirror_candidate(cls, mut):
    """(True, reason) when this candidate provably edits an ALREADY-CORRECT line
    for the inferred symptom class. Never raises; unknown -> (False, "")."""
    if cls not in ("cmp_flip", "bool_flip"):
        return False, ""
    try:
        _off, old_line, new_line = mut
    except (TypeError, ValueError):
        return False, ""
    if not isinstance(old_line, str) or not isinstance(new_line, str):
        return False, ""
    try:
        rule = (_default_idiom_mirror if cls == "bool_flip"
                else _positive_predicate_mirror)
        why = rule(old_line, new_line)
    except Exception:                       # a pre-filter bug must never kill a fix
        return False, ""
    return (True, why) if why else (False, "")


# --------------------------------------- 6c. same-line boundary tie-break
# `book_planning_service::_adult_audience_block`: gold is `target_age > 18` ->
# `< 18`, and the menu offered BOTH `< 18` and `<= 18` on that exact line. Both
# reverse the direction, so both survive the class check and the 1B picks between
# them by coin flip (it took `<= 18` twice and scored 0 twice).
#
# The boundary witnesses separate them: at the boundary value the ORIGINAL is
# False, `<` is also False, and `<=` is True. A "wrong direction" symptom names the
# direction, never the inclusivity — so the candidate that ALSO moves the boundary
# is doing a second, undescribed thing. When some same-line variants preserve
# strictness and others do not, the strictness-preserving ones win and the rest are
# eliminated. Never fires when it cannot produce a UNIQUE winner.

def _literal_boundary(node):
    """True when EVERY comparison on this expression tests against a numeric
    literal. See `_strictness_preserved` for why that gate exists."""
    seen = False
    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            for c in n.comparators:
                if not (isinstance(c, ast.Constant)
                        and isinstance(c.value, (int, float))
                        and not isinstance(c.value, bool)):
                    return False
                seen = True
    return seen


def _strictness_preserved(old_line, new_line):
    """True/False when decidable, None when this is not a same-line cmp variant.

    SCOPE LIMIT, stated plainly because it is the whole reason this is narrow:
    `old < X` -> `> X` and `old < X` -> `>= X` are STRUCTURALLY indistinguishable.
    Both reverse the direction; one preserves strictness, one is the complement,
    and the measured corpus contains a gold of EACH kind:

      * `_adult_audience_block`: `target_age > 18`, gold `< 18` (strict-preserving)
      * `compute_discount`     : `total < threshold`, gold `>= threshold` (complement)

    So there is NO sound universal rule here, and pretending otherwise would just
    trade one wrong answer for another. The tie-break therefore only speaks when the
    boundary is a NUMERIC LITERAL — a concrete value whose inclusivity the symptom
    prose would have had to mention — and stays silent for a symbolic threshold,
    where the model keeps the tie exactly as before. Widen this only with a
    measurement that separates the two cases, never with an argument."""
    old_node, _ = extract_condition(old_line)
    new_node, _ = extract_condition(new_line)
    if old_node is None or new_node is None:
        return None
    if classify_shape(old_node, new_node) != "cmp_op_change":
        return None
    if not (_literal_boundary(old_node) and _literal_boundary(new_node)):
        return None                         # symbolic threshold -> no opinion
    a, b = _cmp_ops(old_node), _cmp_ops(new_node)
    if not a or len(a) != len(b):
        return None
    for oa, ob in zip(a, b):
        if oa is ob:
            continue
        if (oa in _STRICT_OPS) != (ob in _STRICT_OPS):
            return False
    return True


# Words a symptom uses when it is talking about INCLUSIVITY — the boundary VALUE
# itself, not the direction of the test. Only when one of these appears does the
# harness have any business preferring one strictness over the other.
_INCLUSIVITY_RE = re.compile(
    r"\binclusiv\w*\b|\bexclusiv\w*\b|\bor equal\b|\bequal to\b|\bequals\b"
    r"|\boff[- ]by[- ]one\b|\bat the boundary\b|\bboundary value\b"
    r"|\bedge case\b|\bincludes? the\b|\bexcludes? the\b", re.I)


def symptom_names_inclusivity(task_text):
    """True iff the symptom prose actually TALKS about inclusivity."""
    return bool(_INCLUSIVITY_RE.search(task_text or ""))


def _boundary_tiebreak(cls, muts, survivors, evidence, task_text=""):
    """Prefer among same-line operator variants — ONLY when the symptom names
    inclusivity, and then in the direction the symptom points.

    CONTRACT DELIBERATELY INVERTED (stacked autopsy `3de6826bbea7`, ranked fix
    #2). This rule used to eliminate, unconditionally, whichever same-line
    variant CHANGED strictness. On `api/code_submissions.py::_update_lesson_
    progress` that deleted the gold: the fix is `score < 70` -> `score >= 70`,
    and the rule kept `> 70` — wrong at exactly one input (score == 70), which is
    the off-by-one this entire corpus is built to catch. A `<` -> `>=` flip IS
    the canonical boundary fix; a mechanism that treats it as suspect is aimed
    directly at the answer on a `cmp_flip` board.

    The old justification — "a wrong direction symptom names the direction, never
    the inclusivity, so the inclusivity-moving candidate is doing an undescribed
    second thing" — is exactly right ONLY when read the other way round: silence
    about inclusivity is silence, not evidence against it. So the rule now speaks
    only when the symptom SAYS inclusivity, and then it prefers the candidates
    that change inclusivity. Otherwise the tie is real and the model keeps it.
    """
    if cls != "cmp_flip" or len(survivors) < 2:
        return survivors, evidence
    if not symptom_names_inclusivity(task_text):
        return survivors, evidence          # no opinion: the tie is genuine
    surv_ids = {id(m) for m in survivors}
    by_line = {}
    for i, mut in enumerate(muts or []):
        if id(mut) not in surv_ids:
            continue
        try:
            off, old_line, new_line = mut
        except (TypeError, ValueError):
            continue                        # malformed candidate: no opinion
        try:
            verdict = _strictness_preserved(old_line, new_line)
        except Exception:
            verdict = None
        by_line.setdefault(off, []).append((i, mut, verdict))
    doomed = set()
    for _off, group in by_line.items():
        if len(group) < 2:
            continue
        # The symptom names inclusivity => the candidate that MOVES the boundary
        # is the on-target one. (`verdict is False` == strictness changed.)
        keep = [g for g in group if g[2] is False]
        drop = [g for g in group if g[2] is True]
        # Only a UNIQUE winner is a code decision. Anything else stays a tie and
        # the model still chooses.
        if len(keep) != 1 or not drop:
            continue
        for i, mut, _v in drop:
            doomed.add(i)
    if not doomed:
        return survivors, evidence
    out_evidence = []
    for i, ev in enumerate(evidence):
        if i in doomed:
            ev = dict(ev)
            ev["survived"] = False
            ev["boundary_tiebreak"] = False
            ev["reason"] = ("%s | boundary tie-break: the symptom names "
                            "inclusivity and a same-line variant MOVES the "
                            "boundary, so this direction-only variant leaves the "
                            "described defect in place"
                            % ev.get("reason", "")).strip(" |")
        out_evidence.append(ev)
    out_survivors = [m for i, m in enumerate(muts or [])
                     if id(m) in surv_ids and i not in doomed]
    if not out_survivors:                   # unreachable; fail-open law wins
        return survivors, evidence
    return out_survivors, out_evidence


def filter_candidates(task_text, span_lines, muts, cfg=None):
    """Replay every candidate predicate against derived boundary witnesses and
    return the survivors plus per-candidate evidence.

    survivors : subset of `muts`, original order preserved.
    evidence  : index-aligned list of
                {index, line, supported, rows, rows_flipped, delta, shape,
                 survived, reason}
    cfg       : OPTIONAL GymConfig. Absent (the default) or with
                `facts.intent_oracle` off, this function behaves EXACTLY as it did
                before the oracle existed — the knob guard is the only added
                instruction. With the knob on, surviving rows additionally carry
                `intent_oracle` (bool) and `intent_line` (a short menu-ready
                verdict string), and candidates that contradict the function's own
                docstring contract are removed. See intent_oracle.py.

    Contract: UNSUPPORTED ALWAYS SURVIVES. The only eliminations are
      (1) an empty truth-table delta (the candidate provably changes nothing), and
      (2) an AST shape that contradicts the inferred symptom class.
    """
    cls = infer_symptom_class(task_text)
    survivors, evidence = [], []
    for i, mut in enumerate(muts or []):
        # PRE-FILTER (fix #3): a candidate whose BEFORE state already reads the way
        # the symptom describes is a mirror and must not reach the menu at all.
        mirror, why = is_mirror_candidate(cls, mut)
        if mirror:
            ev = {"line": mut[0], "supported": True, "rows": 0, "rows_flipped": 0,
                  "delta": "mirror", "shape": None, "survived": False,
                  "symptom_class": cls, "family_witness": None,
                  "mirror_prefilter": True, "reason": why, "index": i}
            evidence.append(ev)
            continue
        ev = _judge(cls, mut, span_lines)
        ev["index"] = i
        evidence.append(ev)
        if ev["survived"]:
            survivors.append(mut)
    # --- boundary tie-break (fix #6): same-line operator variants -------------
    survivors, evidence = _boundary_tiebreak(cls, muts or [], survivors, evidence,
                                             task_text)
    # --- ADDITIVE: intent oracle (knob facts.intent_oracle, default OFF) ------
    # Byte-identical when the knob is off: cfg defaults to None, so the guard is
    # a single falsy read and nothing below it runs.
    if cfg is not None and _intent_oracle_enabled(cfg) and len(survivors) > 1:
        survivors, evidence = _apply_intent_oracle(
            span_lines, muts or [], survivors, evidence)
    return survivors, evidence


def _judge(cls, mut, span_lines=()):
    base = {"line": None, "supported": False, "rows": 0, "rows_flipped": 0,
            "delta": "unknown", "shape": None, "survived": True,
            "symptom_class": cls, "family_witness": None, "reason": ""}
    try:
        line_idx, old_line, new_line = mut
    except (TypeError, ValueError):
        base["reason"] = "unrecognized candidate shape"
        return base
    base["line"] = line_idx
    # COORDINATED EDIT-SET runs FIRST OF ALL: its `old_line` is a multi-line
    # BLOCK, so every judge below it (which reads one line as one expression)
    # would either abstain or, worse, read the block's first line out of
    # context. `editset_witness` composes the WHOLE set and judges the composed
    # function, which is the only judgement that is true of a set.
    if getattr(mut, "line_span", None) is not None:
        ok, why = editset_witness(span_lines, mut)
        base.update(supported=True, family_witness=ok, survived=ok, reason=why,
                    shape="edit_set")
        return base
    # Contact-family structural witness runs FIRST: it is the only judge that can
    # speak about a guard insertion, an await wrap, a kwarg passthrough or a
    # dominance-checked rename. A verdict here is authoritative in both
    # directions; "does not apply" falls through to the boolean machinery.
    applies, ok, why = family_witness(span_lines, mut)
    if applies:
        base.update(supported=True, family_witness=ok, survived=ok, reason=why,
                    shape="contact_family")
        return base
    try:
        old_node, _ = extract_condition(old_line)
        new_node, _ = extract_condition(new_line)
    except Unsupported as exc:
        base["reason"] = f"unsupported: {exc}"
        return base
    if old_node is None or new_node is None:
        base["reason"] = "unsupported: no condition expression on the line"
        return base
    if not (_is_boolean_shaped(old_node) or _is_boolean_shaped(new_node)):
        base["reason"] = "unsupported: not a boolean/comparison candidate"
        return base
    try:
        keys = free_variables(old_node, new_node)
        rows = witness_rows(keys)
        t_old = truth_table(old_node, rows)
        t_new = truth_table(new_node, rows)
    except Unsupported as exc:
        base["reason"] = f"unsupported: {exc}"
        return base

    flipped = [j for j, (a, b) in enumerate(zip(t_old, t_new)) if a != b]
    shape = classify_shape(old_node, new_node)
    base.update(supported=True, rows=len(rows), rows_flipped=len(flipped),
                shape=shape,
                delta=("empty" if not flipped else
                       "complement" if len(flipped) == len(rows) else "partial"))
    if not flipped:
        base["survived"] = False
        base["reason"] = ("no behavior change over %d boundary witnesses "
                          "(a mirror, not a fix)" % len(rows))
        return base
    if cls is None:
        base["reason"] = ("changes %d/%d witness rows; no symptom class inferable"
                          % (len(flipped), len(rows)))
        return base
    ok, why = _matches_class(cls, shape, old_node, new_node)
    base["survived"] = ok
    base["reason"] = ("%s (%d/%d rows flipped)" % (why, len(flipped), len(rows))) \
        if ok else why
    return base
