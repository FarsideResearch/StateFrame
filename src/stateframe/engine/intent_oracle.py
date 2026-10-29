"""INTENT ORACLE — executable expectations mined from a function's own prose.

WHY THIS EXISTS (E49/E50)
-------------------------
The fix stage is at its own arithmetic ceiling: per-menu accuracy 62.5% file /
83.3% function / 52.8% mutation, product ~27.5%, observed 25.4%. The mutation
menu is the worst rung and witness replay cannot fix it, because witness replay
judges a candidate against BOUNDARY WITNESSES ONLY: every cmp-flip sibling
(`x > y` -> `x < y`, `x >= y`, `x <= y`) reverses the direction, changes rows,
and matches the symptom class. They ALL survive. Nothing in the sandbox encodes
what the function is SUPPOSED to do, so nothing can discriminate.

But intent is usually sitting right there in the function:

    def _compute_age(birth_date, today):
        \"\"\"Return the age in completed years.\"\"\"

    def check_sqs_healthy(depth, threshold):
        \"\"\"Returns True if the queue depth is below the threshold.\"\"\"

"completed years" and "True if X is below Y" are EXECUTABLE claims. This module
turns such prose (plus the signature) into concrete calls with concrete
assertions, runs them against each candidate repair, and kills the candidates
that contradict the function's own stated contract. Zero model calls.

THE THREE LAWS (in priority order)
----------------------------------
1. FAIL OPEN, ALWAYS. Zero expectations extracted, an unbuildable input, an
   exception anywhere -> the candidate list is returned UNTOUCHED. This filter
   may only remove candidates it can positively prove contradict the prose.
2. NO SIGNAL IS NOT A VERDICT. An expectation that the ORIGINAL (buggy) function
   also fails proves nothing — the expectation may simply be wrong, or unbuildable
   for this signature. A candidate may only be killed by an expectation that at
   least one SIBLING candidate passes. That makes every kill a DISCRIMINATION
   between candidates, never a judgement in a vacuum. It also makes it impossible
   to empty the survivor list: the passing sibling always survives.
3. TIERS BOUND THE POWER. EXAMPLE > EXPLICIT_CONDITIONAL > DIRECTIONAL >
   NAME_HINT. A NAME_HINT (`is_*` returns bool) is a smell, not a contract:
   it may only produce a rank_hint (demotion), NEVER a kill.

Deliberately NOT done: inferring intent from the code itself (that is circular —
the code is the thing under repair), and any expectation whose truth we would
have to guess. An expectation we cannot build is dropped silently; the honest
coverage report over the 21-task bench is in test_intent_oracle_offline.py.

Public contract:
    extract(fn_source)  -> [Expectation]
    discriminate(candidates, fn_source, span_lines) -> {
        survivors, killed, no_signal, rank_hints, expectations, applied}
"""
import ast
import datetime
import math
import re
import sys
import textwrap
from dataclasses import dataclass, field

# ------------------------------------------------------------------ tiers
TIER_EXAMPLE = "EXAMPLE"
TIER_EXPLICIT = "EXPLICIT_CONDITIONAL"
TIER_DIRECTIONAL = "DIRECTIONAL"
TIER_NAME_HINT = "NAME_HINT"
TIERS = (TIER_EXAMPLE, TIER_EXPLICIT, TIER_DIRECTIONAL, TIER_NAME_HINT)
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
KILLING_TIERS = frozenset((TIER_EXAMPLE, TIER_EXPLICIT, TIER_DIRECTIONAL))

MAX_EXPECTATIONS = 12          # per function, highest tier first
INSTRUCTION_BUDGET = 40000     # traced line events per call (halts runaway loops)


class Halt(Exception):
    """Instruction budget exhausted — recorded as an execution error, never a fail."""


# --------------------------------------------------------------- outcomes
@dataclass(frozen=True)
class Outcome:
    """One observed call result. `ok=False` means the call raised."""
    ok: bool
    value: object = None
    exc: str = ""


@dataclass(frozen=True)
class Call:
    args: tuple = ()
    kwargs: tuple = ()          # tuple of (name, value) so Call stays hashable

    def as_kwargs(self):
        return dict(self.kwargs)


@dataclass
class Expectation:
    kind: str
    description: str
    tier: str
    calls: tuple
    check: object               # callable(list[Outcome]) -> bool
    source: str = ""            # the prose/name fragment it came from
    meta: dict = field(default_factory=dict)

    def __repr__(self):         # deterministic, test-friendly
        return "Expectation(%s/%s: %s)" % (self.tier, self.kind, self.description)


# =========================================================== 1. EXTRACTION
def _parse_fn(fn_source):
    """(FunctionDef, dedented source) or (None, None). Never raises."""
    if not isinstance(fn_source, str) or not fn_source.strip():
        return None, None
    src = textwrap.dedent(fn_source)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node, src
    return None, None


def _params(fn):
    """Ordered positional parameter names, and the set with defaults."""
    a = fn.args
    names = [p.arg for p in list(a.posonlyargs) + list(a.args)]
    n_def = len(a.defaults)
    optional = set(names[len(names) - n_def:]) if n_def else set()
    return names, optional


def _annotation_name(p):
    ann = getattr(p, "annotation", None)
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Attribute):
        return ann.attr
    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
        return ann.value.id
    return None


def _param_nodes(fn):
    a = fn.args
    return {p.arg: p for p in list(a.posonlyargs) + list(a.args)}


# ---- default value synthesis (name/annotation driven, deliberately boring) ----
_DATE_WORDS = ("date", "day", "birth", "dob", "today", "now", "when", "at",
               "created", "updated", "since", "until", "start", "end")
_NUM_WORDS = ("count", "num", "size", "len", "age", "limit", "threshold", "max",
              "min", "depth", "index", "idx", "n", "total", "amount", "score",
              "years", "months", "days", "offset", "page", "retries", "seconds")
_SEQ_WORDS = ("items", "list", "rows", "values", "results", "files", "names",
              "records", "entries", "options", "args", "chunks")
_STR_WORDS = ("name", "text", "msg", "message", "path", "url", "key", "slug",
              "title", "label", "email", "token", "id", "status")
_BOOL_WORDS = ("flag", "enabled", "enable", "active", "strict", "force", "debug")

_ANN_DEFAULTS = {"int": 1, "float": 1.0, "str": "x", "bool": True,
                 "list": [1], "dict": {}, "set": set(), "tuple": (1,),
                 "date": datetime.date(2020, 1, 1),
                 "datetime": datetime.datetime(2020, 1, 1)}


class _Loose:
    """Stand-in for `self`/an opaque object. Any attribute access yields another
    _Loose, so a method body reaches its arithmetic before it reaches a NameError.
    Comparisons deliberately raise, which becomes an execution error (no signal)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Loose()

    def __repr__(self):
        return "<loose>"


def _default_for(name, ann):
    if ann in _ANN_DEFAULTS:
        v = _ANN_DEFAULTS[ann]
        return list(v) if isinstance(v, list) else v
    low = name.lower()
    if low in ("self", "cls"):
        return _Loose()
    if any(w in low for w in _DATE_WORDS):
        return datetime.date(2020, 1, 1)
    if any(w == low or w in low for w in _SEQ_WORDS):
        return [1]
    if any(w == low or w in low for w in _NUM_WORDS):
        return 1
    if any(w in low for w in _BOOL_WORDS):
        return True
    if any(w in low for w in _STR_WORDS):
        return "x"
    return 1


def _base_binding(fn):
    """{param: default} for every positional parameter."""
    names, _optional = _params(fn)
    nodes = _param_nodes(fn)
    return {n: _default_for(n, _annotation_name(nodes[n])) for n in names}


def _call_from(binding, order):
    return Call(args=tuple(binding[n] for n in order), kwargs=())


# ------------------------------------------------------- docstring plumbing
def _docstring(fn):
    try:
        return ast.get_docstring(fn) or ""
    except Exception:
        return ""


def _sentences(doc):
    out = []
    for chunk in re.split(r"(?<=[.;])\s+|\n", doc or ""):
        s = chunk.strip()
        if s:
            out.append(s)
    return out


def _literal(text):
    """ast.literal_eval with a tolerant fallback for bare True/False/None."""
    t = (text or "").strip().rstrip(".").strip()
    if not t:
        return False, None
    try:
        return True, ast.literal_eval(t)
    except Exception:
        return False, None


# ---------------------------------------------------------- (a) EXAMPLES
_DOCTEST_CALL = re.compile(r"^\s*>>>\s*(?:\w+\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)\s*$")
_PROSE_EXAMPLE = re.compile(
    r"(?:e\.g\.|for example|example:)?\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*"
    r"(?:->|=>|==|returns?|gives?|yields?|is)\s*([^.,;]+)", re.I)


def _example_expectations(fn, doc):
    """Doctest pairs and prose `f(2) -> 3` pairs. Highest tier: an example is the
    author stating the answer, so a candidate contradicting it is simply wrong."""
    out = []
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        m = _DOCTEST_CALL.match(line)
        if not m or m.group(1) != fn.name:
            continue
        expected_src = ""
        for nxt in lines[i + 1:]:
            if nxt.strip():
                expected_src = nxt.strip()
                break
        if expected_src.startswith(">>>"):
            continue
        exp = _build_example(fn, m.group(2), expected_src, ">>> " + line.strip())
        if exp:
            out.append(exp)
    for sent in _sentences(doc):
        if sent.lstrip().startswith(">>>"):
            continue
        m = _PROSE_EXAMPLE.search(sent)
        if not m or m.group(1) != fn.name:
            continue
        exp = _build_example(fn, m.group(2), m.group(3), sent)
        if exp:
            out.append(exp)
    return out


_ARROW_EXAMPLE = re.compile(
    r"""(?:e\.g\.|for example|example:)[^\n]*?
        (?P<in>"[^"]*"|'[^']*'|\[[^\]]*\]|-?\d+(?:\.\d+)?)\s*
        (?:->|=>|→|⇒)\s*
        (?P<out>"[^"]*"|'[^']*'|\[[^\]]*\]|\{[^}]*\}|-?\d+(?:\.\d+)?
                |True|False|None)""",
    re.X)


def _arrow_example_expectations(fn, doc):
    """`e.g. "3" -> ["3", "3.0"]` — an example pair written WITHOUT the function
    name, which is how most humans write them. Requires exactly one non-self
    parameter: with two, the prose does not say which argument the input is, and
    guessing would manufacture a false kill."""
    names, _ = _params(fn)
    real = [n for n in names if n.lower() not in ("self", "cls")]
    if len(real) != 1:
        return []
    out = []
    for m in _ARROW_EXAMPLE.finditer(doc or ""):
        ok_i, arg = _literal(m.group("in"))
        ok_o, expected = _literal(m.group("out"))
        if not (ok_i and ok_o):
            continue
        binding = dict(_base_binding(fn))
        binding[real[0]] = arg

        def check(outs, _e=expected):
            return outs[0].ok and _same(outs[0].value, _e)

        out.append(Expectation(
            kind="arrow_example", tier=TIER_EXAMPLE,
            description="%r -> %r" % (arg, expected),
            calls=(_call_from(binding, names),), check=check,
            source=m.group(0)))
        break                               # the first pair is the load-bearing one
    return out


def _build_example(fn, args_src, expected_src, source):
    ok_e, expected = _literal(expected_src)
    if not ok_e:
        return None
    try:
        call_node = ast.parse("f(%s)" % args_src, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(call_node, ast.Call):
        return None
    args, kwargs = [], []
    try:
        for a in call_node.args:
            args.append(ast.literal_eval(a))
        for k in call_node.keywords:
            if k.arg is None:
                return None
            kwargs.append((k.arg, ast.literal_eval(k.value)))
    except Exception:
        return None
    names, optional = _params(fn)
    supplied = len(args) + len(kwargs)
    required = len([n for n in names if n not in optional])
    if supplied < required or supplied > len(names):
        return None                      # the example does not fit this signature

    def check(outs, _expected=expected):
        o = outs[0]
        return o.ok and _same(o.value, _expected)

    return Expectation(
        kind="example", tier=TIER_EXAMPLE, source=source,
        description="%s(%s) == %r" % (fn.name, args_src.strip(), expected),
        calls=(Call(args=tuple(args), kwargs=tuple(kwargs)),), check=check)


def _same(a, b):
    """Value equality that refuses to conflate True with 1 (the classic silent
    pass in a boolean fix). Numbers of the same kind compare with a tolerance."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return False
    try:
        return a == b
    except Exception:
        return False


# --------------------------------------- (b) explicit conditional returns
_RETURNS = re.compile(
    r"\breturns?\s+(?P<val>True|False|None|-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\")"
    r"\s+(?:if|when|iff|whenever)\s+(?P<cond>.+)$", re.I)
_RAISES = re.compile(
    # deliberately NOT re.I: the exception name must stay capital-initial, so the
    # verb is spelled out both ways instead of loosening the whole pattern.
    r"\b[Rr]aises?\s+(?P<exc>[A-Z]\w*)\s+(?:if|when|whenever|[Ii]f|[Ww]hen)\s+"
    r"(?P<cond>.+)$")

# comparative vocabulary -> (direction, strict) where direction is the truth of
# lhs <op> rhs. "older than" is a real comparison in the domain we care about.
_COMPARATIVES = (
    (r"\b(?:is\s+)?(?:strictly\s+)?greater than\b", "gt", True),
    (r"\b(?:is\s+)?more than\b", "gt", True),
    (r"\bexceeds?\b", "gt", True),
    (r"\bolder than\b", "gt", True),
    (r"\blarger than\b", "gt", True),
    (r"\bnewer than\b", "gt", True),
    (r"\bafter\b", "gt", True),
    (r"\babove\b", "gt", True),
    (r"\b(?:is\s+)?less than\b", "lt", True),
    (r"\bfewer than\b", "lt", True),
    (r"\byounger than\b", "lt", True),
    (r"\bsmaller than\b", "lt", True),
    (r"\bolder\b", "gt", True),
    (r"\bbefore\b", "lt", True),
    (r"\bbelow\b", "lt", True),
    (r"\bunder\b", "lt", True),
    (r"\bat least\b", "ge", False),
    (r"\bno less than\b", "ge", False),
    (r"\bat most\b", "le", False),
    (r"\bno more than\b", "le", False),
)
_COMPARATIVES_RE = tuple((re.compile(p, re.I), d, s) for p, d, s in _COMPARATIVES)

_WORD = re.compile(r"[A-Za-z_]\w*")


def _match_param(phrase, names):
    """Map a noun phrase onto a parameter name. Exact token match wins; then a
    param whose name contains a phrase word (or vice versa). Ambiguity -> None,
    because a wrong binding would manufacture a false kill."""
    words = [w.lower() for w in _WORD.findall(phrase or "")]
    if not words:
        return None
    lowered = {n.lower(): n for n in names}
    for w in words:
        if w in lowered:
            return lowered[w]
    hits = []
    for n in names:
        ln = n.lower().strip("_")
        for w in words:
            if len(w) < 3:
                continue
            if w in ln.split("_") or ln in w or (len(ln) >= 3 and ln in w):
                hits.append(n)
                break
    uniq = sorted(set(hits))
    return uniq[0] if len(uniq) == 1 else None


def _split_comparative(cond):
    """(lhs_phrase, direction, strict, rhs_phrase) or None."""
    for rx, direction, strict in _COMPARATIVES_RE:
        m = rx.search(cond)
        if m:
            return cond[:m.start()], direction, strict, cond[m.end():]
    return None


def _numeric_pair(direction, strict):
    """(satisfying_lhs, violating_lhs, rhs) for lhs <direction> rhs at rhs=5."""
    rhs = 5
    if direction == "gt":
        return 7, 3, rhs
    if direction == "lt":
        return 3, 7, rhs
    if direction == "ge":
        return 5, 4, rhs           # boundary IS satisfying for >=
    if direction == "le":
        return 5, 6, rhs
    return None


def _bind_comparative(fn, cond):
    """Two full bindings for a comparative condition: one satisfying, one
    violating. Returns (sat, viol, description) or None.

    Two shapes are supported: param-vs-param ("depth is below the threshold") and
    param-vs-literal ("count is at least 3"). Anything else is dropped."""
    split = _split_comparative(cond)
    if split is None:
        return None
    lhs_phrase, direction, strict, rhs_phrase = split
    names, _opt = _params(fn)
    names = [n for n in names if n.lower() not in ("self", "cls")]
    if not names:
        return None
    lhs = _match_param(lhs_phrase, names)
    if lhs is None:
        return None
    base = _base_binding(fn)
    pair = _numeric_pair(direction, strict)
    if pair is None:
        return None
    sat_v, viol_v, rhs_default = pair

    rhs = _match_param(rhs_phrase, names)
    if rhs is not None and rhs != lhs:
        sat, viol = dict(base), dict(base)
        sat[lhs], sat[rhs] = sat_v, rhs_default
        viol[lhs], viol[rhs] = viol_v, rhs_default
        desc = "%s %s %s" % (lhs, direction, rhs)
    else:
        ok_n, lit = _literal((_WORD.sub("", rhs_phrase) or "").strip() or
                             (re.search(r"-?\d+(?:\.\d+)?", rhs_phrase or "").group(0)
                              if re.search(r"-?\d+(?:\.\d+)?", rhs_phrase or "")
                              else ""))
        if not ok_n or not isinstance(lit, (int, float)) or isinstance(lit, bool):
            return None
        shift = {"gt": (lit + 2, lit - 2), "lt": (lit - 2, lit + 2),
                 "ge": (lit, lit - 1), "le": (lit, lit + 1)}[direction]
        sat, viol = dict(base), dict(base)
        sat[lhs], viol[lhs] = shift
        desc = "%s %s %r" % (lhs, direction, lit)
    return sat, viol, desc


def _returns_conditional_expectations(fn, doc):
    """"Returns X if/when Y" -> assert X on a Y-satisfying call, and (when X is a
    bool) NOT-X on a Y-violating call. This is the expectation that kills an
    inverted health check."""
    out = []
    names, _ = _params(fn)
    order = [n for n in names]
    for sent in _sentences(doc):
        m = _RETURNS.search(sent)
        if not m:
            continue
        ok_v, expected = _literal(m.group("val"))
        if not ok_v:
            continue
        bound = _bind_comparative(fn, m.group("cond"))
        if bound is None:
            continue
        sat, viol, desc = bound
        calls = (_call_from(sat, order), _call_from(viol, order))
        if isinstance(expected, bool):
            def check(outs, _e=expected):
                if not (outs[0].ok and outs[1].ok):
                    return False
                return _same(bool(outs[0].value), _e) and \
                    _same(bool(outs[1].value), not _e)
            description = "returns %r when %s, %r otherwise" % (
                expected, desc, not expected)
        else:
            calls = calls[:1]

            def check(outs, _e=expected):
                return outs[0].ok and _same(outs[0].value, _e)
            description = "returns %r when %s" % (expected, desc)
        out.append(Expectation(kind="returns_conditional", tier=TIER_EXPLICIT,
                               description=description, calls=calls, check=check,
                               source=sent))
    return out


def _raises_conditional_expectations(fn, doc):
    """"Raises E if Y" -> the Y-satisfying call must raise E (by name)."""
    out = []
    names, _ = _params(fn)
    for sent in _sentences(doc):
        m = _RAISES.search(sent)
        if not m:
            continue
        bound = _bind_comparative(fn, m.group("cond"))
        if bound is None:
            continue
        sat, _viol, desc = bound
        exc_name = m.group("exc")

        def check(outs, _n=exc_name):
            return (not outs[0].ok) and outs[0].exc == _n

        out.append(Expectation(
            kind="raises_conditional", tier=TIER_EXPLICIT,
            description="raises %s when %s" % (exc_name, desc),
            calls=(_call_from(sat, names),), check=check, source=sent))
    return out


# --------------------------------------------- (c) directional / boundary
_BOOLY_PREFIXES = ("is_", "has_", "can_", "should_", "was_", "were_", "does_",
                   "check_", "validate_", "allows_", "needs_")


def _is_bool_shaped(fn, doc):
    if any(fn.name.lstrip("_").startswith(p) for p in _BOOLY_PREFIXES):
        return True
    ret = getattr(fn, "returns", None)
    if isinstance(ret, ast.Name) and ret.id == "bool":
        return True
    return bool(re.search(r"\breturns?\s+(?:True|False)\b", doc or "", re.I))


def _directional_expectations(fn, doc):
    """A bare comparative in the prose ("the older of the two", "values above the
    limit are dropped") with no explicit return value. Weaker than
    EXPLICIT_CONDITIONAL and only usable in two disciplined shapes:

      * bool-shaped function -> satisfying call truthy, violating call falsy
      * otherwise            -> the results must at least DIFFER, which still
                                kills a candidate that collapses the branch

    A collapse-kill is safe under law 2: it fires only when a sibling separates.
    """
    out = []
    names, _ = _params(fn)
    booly = _is_bool_shaped(fn, doc)
    for sent in _sentences(doc):
        if _RETURNS.search(sent) or _RAISES.search(sent):
            continue                      # already covered at a higher tier
        bound = _bind_comparative(fn, sent)
        if bound is None:
            continue
        sat, viol, desc = bound
        calls = (_call_from(sat, names), _call_from(viol, names))
        if booly:
            def check(outs):
                return outs[0].ok and outs[1].ok and \
                    bool(outs[0].value) and not bool(outs[1].value)
            desc_full = "truthy when %s, falsy otherwise" % desc
        else:
            def check(outs):
                return outs[0].ok and outs[1].ok and \
                    not _same(outs[0].value, outs[1].value)
            desc_full = "distinguishes %s from its negation" % desc
        out.append(Expectation(kind="directional", tier=TIER_DIRECTIONAL,
                               description=desc_full, calls=calls, check=check,
                               source=sent))
        break                              # one directional claim per function
    return out


_COMPLETED_UNIT = re.compile(
    r"\b(?:completed|full|whole|elapsed)\s+(years|months|days)\b", re.I)
_AGE_YEARS = re.compile(r"\bage\b.*\byears\b|\byears\b.*\bage\b", re.I)

_DATE_PARAM = re.compile(r"birth|dob|born|start|from|since|created", re.I)
_REF_PARAM = re.compile(r"today|now|ref|as_of|asof|current|end|until|to$", re.I)


def _completed_unit_expectations(fn, doc):
    """"age in completed years" is an exact arithmetic contract, and it is the one
    that discriminates a month/day comparison flip: on the day BEFORE the
    anniversary the answer must be N-1, on the day OF it must be N. A flipped
    comparison returns N on both. Requires two date-shaped parameters so the clock
    is ours, never the machine's."""
    if not (_COMPLETED_UNIT.search(doc or "") or _AGE_YEARS.search(doc or "")):
        return []
    unit_m = _COMPLETED_UNIT.search(doc or "")
    unit = (unit_m.group(1).lower() if unit_m else "years")
    if unit != "years":
        return []
    names, _ = _params(fn)
    names = [n for n in names if n.lower() not in ("self", "cls")]
    start = next((n for n in names if _DATE_PARAM.search(n)), None)
    ref = next((n for n in names if _REF_PARAM.search(n)), None)
    if start is None:
        return []
    base = _base_binding(fn)
    before, on = dict(base), dict(base)
    if ref is not None and ref != start:
        # both endpoints are parameters: move the REFERENCE across the anniversary
        born = datetime.date(2000, 6, 15)
        before[start], before[ref] = born, datetime.date(2020, 6, 14)
        on[start], on[ref] = born, datetime.date(2020, 6, 15)
    else:
        # only the start is a parameter; the function reads the clock, which the
        # sandbox has FROZEN (see FROZEN_TODAY). Move the BIRTH date instead —
        # this is the shape `_compute_age(dob)` actually has in the wild.
        before[start] = datetime.date(FROZEN_TODAY.year - 20,
                                      FROZEN_TODAY.month, FROZEN_TODAY.day + 1)
        on[start] = datetime.date(FROZEN_TODAY.year - 20,
                                  FROZEN_TODAY.month, FROZEN_TODAY.day)
    order = _params(fn)[0]

    def check(outs):
        if not (outs[0].ok and outs[1].ok):
            return False
        return _same(outs[0].value, 19) and _same(outs[1].value, 20)

    return [Expectation(
        kind="completed_years", tier=TIER_EXPLICIT,
        description="completed years: day-before-anniversary 19, on-anniversary 20",
        calls=(_call_from(before, order), _call_from(on, order)), check=check,
        source=(unit_m.group(0) if unit_m else "age ... years"))]


_EMPTY_DEFAULT = re.compile(
    r"\bempty\b[^.;]*?(?:returns?|->|gives?|yields?)\s*"
    r"(\[\]|\{\}|''|\"\"|0|None|False|True)", re.I)
_NONE_INPUT = re.compile(
    r"\bNone\b[^.;]*?(?:returns?|->|gives?|yields?)\s*"
    r"(\[\]|\{\}|''|\"\"|0|None|False|True)", re.I)


def _boundary_expectations(fn, doc):
    """"empty -> default" / "None -> default": one concrete substituted argument.
    Only fires with exactly ONE non-self parameter, because with two the prose does
    not say WHICH one is empty and a guess would be a false kill."""
    names, _ = _params(fn)
    names = [n for n in names if n.lower() not in ("self", "cls")]
    if len(names) != 1:
        return []
    order = _params(fn)[0]
    base = _base_binding(fn)
    out = []
    for rx, subst, label in ((_EMPTY_DEFAULT, [], "empty"),
                             (_NONE_INPUT, None, "None")):
        m = rx.search(doc or "")
        if not m:
            continue
        ok_v, expected = _literal(m.group(1))
        if not ok_v:
            continue
        binding = dict(base)
        binding[names[0]] = subst

        def check(outs, _e=expected):
            return outs[0].ok and _same(outs[0].value, _e)

        out.append(Expectation(
            kind="boundary_%s" % label, tier=TIER_EXPLICIT,
            description="%s %s -> %r" % (names[0], label, expected),
            calls=(_call_from(binding, order),), check=check, source=m.group(0)))
    return out


# ------------------------------------------------------------ (d) name hints
def _name_hint_expectations(fn, doc):
    """Contracts encoded in the NAME. Rank-only by law 3: `is_x` returning a
    non-bool is a smell, and plenty of correct code is smelly."""
    out = []
    names, _ = _params(fn)
    base = _call_from(_base_binding(fn), names)
    bare = fn.name.lstrip("_")
    if any(bare.startswith(p) for p in ("is_", "has_", "can_", "should_")):
        def check_bool(outs):
            return outs[0].ok and isinstance(outs[0].value, bool)
        out.append(Expectation(
            kind="name_returns_bool", tier=TIER_NAME_HINT,
            description="%s() should return a bool" % fn.name,
            calls=(base,), check=check_bool, source=fn.name))
    if re.search(r"(^|_)(to|as|convert|parse|format|build|make|render)(_|$)", bare):
        def check_not_none(outs):
            return outs[0].ok and outs[0].value is not None
        out.append(Expectation(
            kind="name_converts", tier=TIER_NAME_HINT,
            description="%s() should produce a value, not None" % fn.name,
            calls=(base,), check=check_not_none, source=fn.name))
    if re.search(r"(s|list|all)$", bare) and not bare.endswith("ss"):
        def check_iterable(outs):
            return outs[0].ok and hasattr(outs[0].value, "__iter__")
        out.append(Expectation(
            kind="name_plural_iterable", tier=TIER_NAME_HINT,
            description="%s() should return something iterable" % fn.name,
            calls=(base,), check=check_iterable, source=fn.name))
    return out


# ------------------------------------------------------------------ extract
_EXTRACTORS = (
    _example_expectations,
    _arrow_example_expectations,
    _completed_unit_expectations,
    _returns_conditional_expectations,
    _raises_conditional_expectations,
    _boundary_expectations,
    _directional_expectations,
    _name_hint_expectations,
)


def extract(fn_source):
    """Executable expectations mined from a function's prose + signature.

    Deterministic, no model call, never raises. An empty list is the honest and
    common answer: most functions do not state a checkable contract.
    """
    fn, _src = _parse_fn(fn_source)
    if fn is None:
        return []
    doc = _docstring(fn)
    out = []
    for extractor in _EXTRACTORS:
        try:
            out.extend(extractor(fn, doc) or [])
        except Exception:
            continue                      # one broken miner never blocks the rest
    out.sort(key=lambda e: TIER_RANK.get(e.tier, len(TIERS)))
    return out[:MAX_EXPECTATIONS]


# ============================================================ 2. EXECUTION
# FROZEN CLOCK. Real code says `today = date.today()`, and a wall-clock read makes
# every date expectation non-deterministic (and unrepeatable across cycles). The
# sandbox therefore hands the candidate a `date`/`datetime` whose today()/now() is
# a fixed instant. Subclassing keeps `isinstance(x, date)` and every comparison
# working; nothing else about the type changes.
FROZEN_TODAY = datetime.date(2020, 6, 15)
FROZEN_NOW = datetime.datetime(2020, 6, 15, 12, 0, 0)


class _FrozenDate(datetime.date):
    @classmethod
    def today(cls):
        return datetime.date(FROZEN_TODAY.year, FROZEN_TODAY.month,
                             FROZEN_TODAY.day)


class _FrozenDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW if tz is None else FROZEN_NOW.replace(tzinfo=tz)

    @classmethod
    def utcnow(cls):
        return FROZEN_NOW

    @classmethod
    def today(cls):
        return FROZEN_NOW


class _DatetimeShim:
    """`datetime` module surface, clock frozen. Attribute lookups fall through to
    the real module so anything we did not think about still works."""
    date = _FrozenDate
    datetime = _FrozenDateTime

    def __getattr__(self, name):
        return getattr(datetime, name)


_SAFE_MODULES = {"datetime": _DatetimeShim(), "date": _FrozenDate,
                 "timedelta": datetime.timedelta,
                 "time": datetime.time, "math": math, "re": re}


def _budgeted(limit):
    count = [0]

    def tracer(frame, event, arg):
        count[0] += 1
        if count[0] > limit:
            raise Halt("instruction budget exhausted")
        return tracer
    return tracer


def _load(fn_source, name):
    """Compile+exec the function in an isolated namespace and hand back the
    callable. Any failure -> None, which becomes 'no signal' everywhere upstream."""
    src = textwrap.dedent(fn_source or "")
    ns = dict(_SAFE_MODULES)
    try:
        code = compile(src, "<intent_oracle>", "exec")
        exec(code, ns)                    # noqa: S102 — candidate source, sandboxed ns
    except Exception:
        return None
    fn = ns.get(name)
    return fn if callable(fn) else None


def run_call(fn, call):
    """One observed outcome. Never raises; a runaway loop is halted by budget."""
    if fn is None:
        return Outcome(ok=False, exc="Unloadable")
    old = sys.gettrace()
    sys.settrace(_budgeted(INSTRUCTION_BUDGET))
    try:
        value = fn(*call.args, **call.as_kwargs())
        return Outcome(ok=True, value=value)
    except Halt:
        return Outcome(ok=False, exc="Halt")
    except Exception as exc:
        return Outcome(ok=False, exc=type(exc).__name__)
    except BaseException as exc:          # SystemExit et al from repo code
        return Outcome(ok=False, exc=type(exc).__name__)
    finally:
        sys.settrace(old)


def evaluate(fn_source, name, expectations):
    """{expectation index: True/False/None}. None = could not observe (no signal)."""
    fn = _load(fn_source, name)
    verdicts = {}
    for i, exp in enumerate(expectations):
        if fn is None:
            verdicts[i] = None
            continue
        outs = [run_call(fn, c) for c in exp.calls]
        if any(o.exc in ("Halt", "Unloadable") for o in outs):
            verdicts[i] = None
            continue
        try:
            verdicts[i] = bool(exp.check(outs))
        except Exception:
            verdicts[i] = None
    return verdicts


# ========================================================= 3. DISCRIMINATION
def _apply(span_lines, mut):
    """The candidate's function source, or None when it cannot be applied.

    `mut` line indices are ABSOLUTE file offsets while span_lines is span-relative
    (the same contract witness_replay.family_witness works under), so the anchor is
    the OLD LINE's text. Ambiguity -> None: a wrong anchor could manufacture a
    false kill, and law 1 says we would rather learn nothing."""
    try:
        _idx, old_line, new_line = mut
    except (TypeError, ValueError):
        return None
    if not isinstance(old_line, str) or not isinstance(new_line, str):
        return None
    hits = [i for i, line in enumerate(span_lines) if line == old_line]
    if len(hits) != 1:
        return None
    out = list(span_lines)
    out[hits[0]] = new_line
    src = "\n".join(out)
    try:
        ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
    return src


def discriminate(candidates, fn_source, span_lines):
    """Kill the candidates that contradict the function's own stated intent.

    Returns {survivors, killed, no_signal, rank_hints, expectations, applied}:
      survivors  : subset of `candidates`, ORIGINAL ORDER PRESERVED
      killed     : [(candidate, expectation, reason)]
      no_signal  : [(expectation, reason)] — expectations that decided nothing
      rank_hints : [(candidate, expectation, reason)] — NAME_HINT demotions only

    FAIL-OPEN: on zero expectations, an unbuildable candidate, or ANY exception,
    the full candidate list is returned untouched.
    """
    cands = list(candidates or [])
    empty = {"candidates": cands, "survivors": cands, "killed": [],
             "passes": [], "no_signal": [], "rank_hints": [],
             "expectations": [], "applied": False}
    if len(cands) < 2:
        return empty                     # nothing to discriminate BETWEEN
    try:
        return _discriminate(cands, fn_source, list(span_lines or []), empty)
    except Exception:
        return empty


def _discriminate(cands, fn_source, span_lines, empty):
    fn, _src = _parse_fn(fn_source)
    if fn is None:
        return empty
    expectations = extract(fn_source)
    if not expectations:
        return empty

    # The ORIGINAL (still-buggy) function is the control arm: an expectation it
    # already fails proves nothing about any candidate (law 2).
    base_verdicts = evaluate(fn_source, fn.name, expectations)

    sources = [_apply(span_lines, m) for m in cands]
    verdicts = [evaluate(s, fn.name, expectations) if s is not None else None
                for s in sources]

    killed, rank_hints, no_signal, passes = [], [], [], []
    doomed = set()
    for ei, exp in enumerate(expectations):
        passers = [ci for ci, v in enumerate(verdicts)
                   if v is not None and v.get(ei) is True]
        failers = [ci for ci, v in enumerate(verdicts)
                   if v is not None and v.get(ei) is False]
        if not passers:
            no_signal.append((exp, "no candidate satisfies it (base=%r)"
                              % base_verdicts.get(ei)))
            continue
        for ci in passers:               # recorded even when nothing is killed,
            passes.append((cands[ci], exp))   # so the menu can show the positive
        if not failers:
            no_signal.append((exp, "every observable candidate satisfies it"))
            continue
        if exp.tier not in KILLING_TIERS:
            for ci in failers:
                rank_hints.append((cands[ci], exp,
                                   "name hint not met: %s" % exp.description))
            continue
        for ci in failers:
            doomed.add(ci)
            killed.append((cands[ci], exp,
                           "contradicts stated intent: %s" % exp.description))

    survivors = [c for ci, c in enumerate(cands) if ci not in doomed]
    if not survivors:                    # Unreachable by construction; retained as a defensive check.
        return empty
    return {"candidates": cands, "survivors": survivors, "killed": killed,
            "passes": passes, "no_signal": no_signal, "rank_hints": rank_hints,
            "expectations": expectations, "applied": bool(killed or rank_hints)}


# -------------------------------------------------- 4. MENU EVIDENCE LINES
# Design rule: the model must confirm a
# pre-digested verdict, never guess. A filter that silently drops options teaches
# the model nothing; the same judgement rendered ON the option turns the menu into
# a reading task. These lines are pure strings — the caller decides whether to show
# them, and showing them is knob-routed exactly like the filtering.
EVIDENCE_MAX_CHARS = 96          # menu option tails must stay one short line


def _clip(text, limit=EVIDENCE_MAX_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def render_evidence(results):
    """One short evidence line per candidate, index-aligned with
    `results["candidates"]` (original menu order). Pure: no execution, no I/O.

    Three renderable outcomes, in precedence order:
      FAILS      — a killing-tier expectation the candidate provably contradicts
      weak       — a NAME_HINT it misses (demotion only, the option still stands)
      passes     — a stated contract it satisfies while a sibling does not
      no signal  — nothing decided; the model is on its own for this option
    """
    if not isinstance(results, dict):
        return []
    cands = list(results.get("candidates") or [])
    if not cands:
        return []

    def index_of(cand):
        for i, c in enumerate(cands):
            if c is cand or c == cand:
                return i
        return None

    lines = ["no signal"] * len(cands)
    for cand, exp in (results.get("passes") or []):
        i = index_of(cand)
        if i is not None and lines[i] == "no signal":
            lines[i] = _clip("passes docstring check: '%s'" % exp.description)
    for cand, exp, _why in (results.get("rank_hints") or []):
        i = index_of(cand)
        if i is not None:
            lines[i] = _clip("weak: %s" % exp.description)
    for cand, exp, _why in (results.get("killed") or []):
        i = index_of(cand)
        if i is not None:
            lines[i] = _clip("FAILS docstring check: '%s'" % exp.description)
    return lines


# ------------------------------------------------------------- coverage aid
def coverage(fn_sources):
    """{tier: count of sources yielding >=1 expectation at that tier} plus totals.
    Used by the offline bench report — no model, no side effects."""
    sources = list(fn_sources or [])
    per_tier = {t: 0 for t in TIERS}
    any_count = killing = 0
    for src in sources:
        exps = extract(src)
        tiers = {e.tier for e in exps}
        if exps:
            any_count += 1
        if tiers & KILLING_TIERS:
            killing += 1
        for t in tiers:
            per_tier[t] = per_tier.get(t, 0) + 1
    return {"total": len(sources), "with_any": any_count,
            "with_killing_tier": killing, "per_tier": per_tier}
