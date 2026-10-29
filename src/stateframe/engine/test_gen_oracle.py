"""TEST-GENERATION ORACLE — the deterministic tie-breaker at the mutation menu.

THE GAP THIS FILLS
------------------
When two or more candidate fixes both compile and survive every existing gate,
nothing external decides. The margin probe is a coin flip, the ballot is thrown
away, and a 1B model picks a sibling line. Measured on the gym bench: the gold
line is IN the menu 91% of the time and picked 43% of the time. Discrimination,
not retrieval, is the bottleneck.

This module synthesizes DISCRIMINATING TEST CASES from AST facts + the mutation
family, executes every candidate AND the original against them in an isolated
namespace with a frozen clock and an instruction budget, and reports which
candidate — if any — is uniquely consistent with the symptom's stated direction.

THE THREE HONESTY RULES (from the committed design, sol_answers/w8/a_test_synth.md)
1. DISCRIMINATION.  A synthesized case is USEFUL only if at least two candidates
   disagree on it. A case everyone agrees on teaches nothing and is not counted
   as evidence.
2. VACUITY SELF-DETECTION.  A case on which the ORIGINAL and every candidate
   agree is VACUOUS: it cannot be a regression test (it would pass pre-fix and
   post-fix alike). Vacuous cases are excluded and COUNTED, never reported as
   passing evidence.
3. FAIL-OPEN.  A predicate that would empty the survivor set is discarded. The
   result is a verdict of AMBIGUOUS or NO_SIGNAL, never a wrong elimination.
   Killing the gold costs an unfixable task; a false survival costs one model
   call. The asymmetry is ~infinite, so the bar is: ZERO gold eliminations.

WHAT IT DOES NOT DO
-------------------
It does not invent ground truth. Every oracle predicate below is derived from
something the function itself already says (its docstring, its raise messages,
its own branch structure). Where the function says nothing checkable, the honest
answer is NO_SIGNAL, and that is the common answer.
"""

import ast
import itertools
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field

import intent_oracle as io
import witness_replay as wr

# --------------------------------------------------------------------- budget
MAX_CASES = 72              # synthesized cases per menu (hard ceiling)
MAX_PROBED_PARAMS = 2       # beyond this the product blows the ceiling
# NEVER TRUNCATE THE MENU. A candidate dropped for budget reasons is reported as
# eliminated by every downstream consumer, and on this bench the real menus reach
# 77 options — a cap of 24 "eliminated" the gold on 5 tasks purely by bookkeeping.
# The budget is spent on CASES instead, which only ever costs resolution.
MAX_EXECUTIONS = 6000       # candidate x case function calls per menu
MIN_CASES = 8               # floor: below this there is nothing to discriminate
INSTRUCTION_BUDGET = 20000  # traced line events per call
VALUE_REPR_CLIP = 160       # outcome fingerprints stay comparable, not huge
#: Bytes one traced call may add to the process RSS before it is halted.
#: MEASURED 2026-08-29: the gym's 4-task real bench (results/real_v3_err4.jsonl)
#: was SIGKILLed (exit 137) with zero rows written. Solo, task 0a492660 alone
#: climbed 0.10 GB -> 12.9 GB in 24s and died; seven faulthandler dumps taken
#: between 0.55 GB and 12.9 GB carried the SAME stack, every frame identical:
#: run_case -> _tracer.trace. One traced call was allocating without bound.
#: An instruction budget cannot see that: a single line event can allocate a
#: gigabyte ("x" * 10**9), so events and bytes are independent budgets and the
#: sandbox needs both.
MEMORY_BUDGET_BYTES = 256 * 1024 * 1024
#: Live CPython allocations one traced call may ADD. Catches the "millions of
#: small objects" runaway, which moves blocks but barely moves RSS.
BLOCK_BUDGET = 8_000_000
#: Seconds between memory samples. THE SAMPLING AXIS IS TIME, NOT EVENTS, and
#: the reason is measured rather than fastidious.
#:
#: The candidate that killed the bench was
#:
#:     for env_var in known:          # was: in _PROVIDER_KEY_ENV_VARS
#:         known.append((f"env:{env_var}", ...))
#:
#: — iterating the list it appends to, where element k embeds element k-1's
#: repr. Growth is EXPONENTIAL per iteration, so the process went from megabytes
#: to 12.9 GB inside one 2048-event sampling gap and the periodic check never
#: ran. Event-count sampling has NO bound on the overshoot: pick any period N,
#: and N doublings defeat it.
#:
#: Sampling on the clock does have a bound. Allocation rate is capped by memory
#: bandwidth (order 10 GB/s), so an interval of `t` seconds bounds the overshoot
#: at roughly `bandwidth * t` — about 50 MB here — no matter what shape the
#: growth is. That is the whole argument for this constant.
#:
#: Checking every event was tried first and MEASURED at 15.6s -> 164s on
#: test_test_gen_oracle_offline.py (getrusage is a syscall; the traced loops make
#: millions of events). `time.monotonic` is not a syscall and is ~25x cheaper.
MEMORY_CHECK_INTERVAL_S = 0.005
#: Wall-clock seconds ONE traced call may run. This is the budget that actually
#: bounds a runaway, and it exists because the memory budget alone provably does
#: not: `ru_maxrss` is a HIGH-WATER MARK that never falls, so a per-call growth
#: delta re-baselines at the new high water on every call and RATCHETS. Measured
#: after the first fix landed: each of ~348 candidate x case calls was correctly
#: halted at +256 MB, and the process still climbed 0.39 GB -> 12.97 GB and died,
#: one legitimate 256 MB step at a time.
#: A deadline has no such hole, costs nothing (the clock is already being read),
#: and is six orders of magnitude above a legitimate probe case, which completes
#: in microseconds — the whole offline board is ~350 calls in under 20 seconds.
CALL_DEADLINE_S = 0.5

SANDBOX_FILENAME = "<test_gen_oracle>"

# Verdicts
SELECTED = "SELECTED"
AMBIGUOUS = "AMBIGUOUS"
NO_SIGNAL = "NO_SIGNAL"

# Case classifications (design Q3, adapted: we have no post-fix truth, so the
# axis is agreement rather than pass/fail).
USEFUL = "DISCRIMINATING"
VACUOUS = "VACUOUS"
INERT = "INERT"             # candidates differ from the original but not from
                            # each other — real evidence the line matters, but
                            # it cannot separate two candidates.

# Boolean-shaped probe ladder, highest-signal first (witness_replay's ordering).
BOOL_PROBES = (None, "", 0, False, [], 1, "x", True, [1])
NUMERIC_FLOOR = (-1, 0, 1, 2)


# ------------------------------------------------------------------- results
@dataclass(frozen=True)
class ProbeCase:
    """One synthesized input, with the template that produced it."""
    template: str           # "cmp_flip:boundary", "none_flip:none", ...
    label: str              # human-readable binding, for citation
    call: object            # intent_oracle.Call

    def __repr__(self):
        return "ProbeCase(%s %s)" % (self.template, self.label)


@dataclass
class DiscriminationResult:
    verdict: str = NO_SIGNAL
    selected: object = None          # the winning candidate, or None
    survivors: list = field(default_factory=list)
    cases: list = field(default_factory=list)          # every ProbeCase built
    discriminating: list = field(default_factory=list)  # indices into .cases
    vacuous: list = field(default_factory=list)         # indices into .cases
    inert: list = field(default_factory=list)           # indices into .cases
    cases_cited: list = field(default_factory=list)     # labels backing the pick
    predicates: list = field(default_factory=list)      # (name, n_consistent)
    reason: str = ""

    @property
    def counts(self):
        return {"cases": len(self.cases), "discriminating": len(self.discriminating),
                "vacuous": len(self.vacuous), "inert": len(self.inert),
                "survivors": len(self.survivors)}


# ==================================================== 1. AST FACT EXTRACTION
def _fn_node(fn_source):
    try:
        tree = ast.parse(textwrap.dedent(fn_source or ""))
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _param_names(fn):
    a = fn.args
    names = [p.arg for p in list(a.posonlyargs) + list(a.args)]
    return [n for n in names if n not in ("self", "cls")]


def _numeric_literals(node):
    """Every int/float constant the function compares or arithmetics against.

    These are the boundaries a cmp_flip/off_by_one candidate moves, so they are
    exactly where two candidates disagree.
    """
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            out.append(n.value)
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq[:6]


def _line_names(line):
    """Identifiers mentioned on a source line (the mutated line's operands)."""
    try:
        node, _is_test = wr.extract_condition(line)
    except Exception:
        node = None
    if node is None:
        try:
            node = ast.parse(textwrap.dedent(line).strip())
        except Exception:
            return []
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                names.append(base.id)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _candidate_family(old_line, new_line):
    """Best-effort family label for ONE candidate, from the two line texts."""
    try:
        old_node, _ = wr.extract_condition(old_line)
        new_node, _ = wr.extract_condition(new_line)
        if old_node is not None and new_node is not None:
            shape = wr.classify_shape(old_node, new_node)
            if shape:
                return shape
    except Exception:
        pass
    if "None" in old_line or "None" in new_line:
        return "none_flip"
    return "unknown"


# ============================================== 2. PER-FAMILY CASE TEMPLATES
# A closed registry. An unknown family gets the union ladder, never a bespoke
# guess: a template that cannot state WHY its inputs discriminate is not a
# template, it is a random smoke test.

def _probe_values(family, param, literals):
    """The value ladder for one parameter under one mutation family."""
    low = param.lower()
    if family == "none_flip":
        # None + a non-None control, so the case proves the guard AND proves
        # normal behaviour is intact (design Q1, none_guard row).
        return [None, io._default_for(param, "")]
    if family in ("cmp_flip", "off_by_one", "arith_flip"):
        vals = list(NUMERIC_FLOOR)
        for k in literals:                       # boundary probes at K-1, K, K+1
            if isinstance(k, float):
                vals.extend([k - 1.0, k, k + 1.0])
            else:
                vals.extend([k - 1, k, k + 1])
        seen, uniq = set(), []
        for v in vals:
            key = (type(v).__name__, v)
            if key not in seen:
                seen.add(key)
                uniq.append(v)
        if any(w in low for w in ("text", "str", "name", "msg", "grade")):
            # a numeric threshold read off a STRING param (`int(grade)`) still
            # needs the boundary, spelled the way the parameter takes it
            uniq = [str(v) for v in uniq if isinstance(v, int)] + uniq
        return uniq[:12]
    if family == "bool_flip":
        return list(BOOL_PROBES)
    return list(BOOL_PROBES) + [k for k in literals]


def _target_params(fn, candidates, params):
    """Which parameters the candidates actually touch.

    Probing a parameter no candidate mentions produces cases where everyone
    agrees by construction — pure vacuity. So the probe set is the intersection
    of the mutated lines' operands with the signature.
    """
    mentioned = []
    for cand in candidates:
        try:
            _idx, old_line, new_line = cand
        except Exception:
            continue
        for name in _line_names(old_line) + _line_names(new_line):
            if name in params and name not in mentioned:
                mentioned.append(name)
    if mentioned:
        return mentioned[:MAX_PROBED_PARAMS]
    return params[:MAX_PROBED_PARAMS]


def synthesize_cases(fn_source, candidates, symptom_class, max_cases=MAX_CASES):
    """The synthesized case list: per-family templates x AST facts.

    Deterministic and bounded. Never raises — an unparseable function yields an
    empty list, which becomes NO_SIGNAL upstream.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return []
    params = _param_names(fn)
    literals = _numeric_literals(fn) + [
        v for v in _resolved_constants(fn_source) if v not in _numeric_literals(fn)]
    base = {}
    try:
        base = io._base_binding(fn)
    except Exception:
        base = {n: io._default_for(n, "") for n in params}
    order = _param_names(fn)
    full_order = [p.arg for p in list(fn.args.posonlyargs) + list(fn.args.args)]

    families = []
    for cand in candidates:
        try:
            fam = _candidate_family(cand[1], cand[2])
        except Exception:
            fam = "unknown"
        if fam not in families:
            families.append(fam)
    if symptom_class and symptom_class not in families:
        families.insert(0, symptom_class)
    if not families:
        families = ["unknown"]

    probed = _target_params(fn, candidates, params)
    cases, seen = [], set()
    for family in families:
        ladders = [_probe_values(family, p, literals) for p in probed]
        if not ladders:
            ladders = [[None]]
            probed_here = []
        else:
            probed_here = probed
        for combo in itertools.product(*ladders):
            if len(cases) >= max_cases:
                return cases
            binding = dict(base)
            for name, value in zip(probed_here, combo):
                binding[name] = value
            args = []
            for name in full_order:
                if name in ("self", "cls"):
                    args.append(io._Loose())
                else:
                    args.append(binding.get(name, io._default_for(name, "")))
            label = ", ".join("%s=%r" % (n, v)
                              for n, v in zip(probed_here, combo)) or "<no args>"
            key = (family, label)
            if key in seen:
                continue
            seen.add(key)
            cases.append(ProbeCase(template="%s:probe" % family, label=label,
                                   call=io.Call(args=tuple(args))))
    return cases


# ==================================================== 3. SANDBOXED EXECUTION
# Same discipline as intent_oracle: isolated namespace seeded with a FROZEN
# clock, a traced instruction budget so a runaway loop cannot hang the kernel,
# and total failure tolerance (anything unloadable is "no signal", never a kill).
# The one addition is LINE COVERAGE, which the dead-branch predicate needs.

def _free_globals(fn_source):
    """Global names the span references but does not define.

    A harvested span is a FUNCTION, not a module: `logging.INFO` resolves at call
    time against a namespace that no longer has `logging` in it. Every case then
    dies of NameError before reaching the mutated line, and the whole menu looks
    unobservable. Re-binding the free names is what makes threshold cascades
    against module constants executable at all.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return []
    bound = set(_param_names(fn)) | {fn.name}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    free = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id not in bound and node.id not in free:
            free.append(node.id)
    return free


def _namespace(fn_source):
    """Isolated namespace: frozen clock, plus a best-effort binding for every
    free global. A real module when the name imports (so `logging.INFO` is the
    genuine 20), an opaque stand-in otherwise. Never executes repo code."""
    import builtins
    import importlib
    ns = dict(io._SAFE_MODULES)
    for name in _free_globals(fn_source):
        if name in ns or hasattr(builtins, name):
            continue
        # STDLIB ONLY. A bare `import_module(name)` would happily import a repo
        # module off sys.path and run its top level — real side effects, from a
        # sandbox whose whole job is not to have any.
        if name in getattr(sys, "stdlib_module_names", frozenset()):
            try:
                ns[name] = importlib.import_module(name)
                continue
            except Exception:
                pass
        ns[name] = _synthetic_constant(name)
    return ns


# An UPPER_CASE free global is almost always a module-level tuning constant
# (`MAX_FILES_PER_SUBMISSION`). Binding it to the opaque stand-in makes EVERY
# comparison against it raise TypeError, which silently blinds every predicate:
# measured on `validate_file_count`, all 40 synthesized cases raised and the
# raise-message contract could not be evaluated at all. A synthetic number keeps
# the comparison observable. It is a probe-space choice, not a truth claim —
# predicates only ever ask about the function's consistency with ITSELF.
_LIMIT_WORDS = ("max", "limit", "cap", "ceiling", "upper", "timeout", "size")
_FLOOR_WORDS = ("min", "floor", "lower", "start", "base")


class _Opaque:
    """A CALLABLE stand-in for an unresolvable global.

    `intent_oracle._Loose` is deliberately not callable, and that costs an entire
    function here: a trailing `logger.info(...)` then raises TypeError on EVERY
    path, so `validate_file_count` produced 40 identical raised outcomes and no
    predicate could see anything. A stub that swallows calls keeps the function's
    own logic observable, which is the only thing being measured.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Opaque()

    def __call__(self, *args, **kwargs):
        return _Opaque()

    def __repr__(self):
        return "<opaque>"


def _synthetic_constant(name):
    """The stand-in for one unresolvable free global, chosen by what the name is.

    An exception name gets a REAL exception class, so `raise NotFound(...)` is a
    clean raise rather than "exceptions must derive from BaseException" noise
    that erases the difference between two candidates' error paths.
    """
    if name.endswith(("Error", "Exception", "Warning")):
        return type(name, (Exception,), {})
    if not name.isupper():
        return _Opaque()
    low = name.lower()
    if any(w in low for w in _LIMIT_WORDS):
        return 100
    if any(w in low for w in _FLOOR_WORDS):
        return 0
    return 1


def _resolved_constants(fn_source):
    """Numeric values of comparison right-hand sides, resolved in the namespace.

    `if levelno >= logging.WARNING` states a boundary at 30 just as loudly as a
    bare literal does; without resolving it the probe ladder never lands near the
    threshold and every case falls in the same branch.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return []
    ns = _namespace(fn_source)
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left] + list(node.comparators):
            if isinstance(side, ast.Constant):
                continue                  # already covered by _numeric_literals
            try:
                value = eval(compile(ast.Expression(side), SANDBOX_FILENAME,
                                     "eval"), dict(ns))   # noqa: S307
            except Exception:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and value not in out:
                out.append(value)
    return out[:6]


def _load(fn_source, name):
    src = textwrap.dedent(fn_source or "")
    ns = _namespace(fn_source)
    try:
        code = compile(src, SANDBOX_FILENAME, "exec")
        exec(code, ns)                # noqa: S102 — candidate source, isolated ns
    except Exception:
        return None
    fn = ns.get(name)
    return fn if callable(fn) else None


class _HardHalt(BaseException):
    """The sandbox budget, in a form candidate code CANNOT swallow.

    MEASURED 2026-08-29 (`test_a_budget_halt_survives_a_candidates_except_clause`).
    `io.Halt` derives from `Exception`, and CPython DISABLES tracing the moment a
    trace function raises. So a candidate whose loop body is wrapped in a bare
    `except Exception:` both catches the halt AND keeps running with the tracer
    gone: the budget is spent, unenforceable, and the loop is now genuinely
    infinite. Production code is full of `except Exception: continue`, so this is
    not a corner case — it is how the real-history bench died.

    A `BaseException` cannot be caught by `except Exception`, so the halt always
    reaches `run_case`, which maps it back to the SAME `exc="Halt"` outcome the
    catchable version produced. Fingerprints are unchanged for every call that
    was already halting correctly.
    """


def _rss_bytes():
    """Process RSS high-water mark in bytes, or None where it cannot be read.

    macOS reports `ru_maxrss` in bytes, Linux in kilobytes. Guessing wrong by
    1024x turns the memory budget into either a no-op or an instant halt, so
    normalise on the platform rather than on a hopeful constant.
    """
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:                       # noqa: BLE001 — budget is optional
        return None
    return raw if sys.platform == "darwin" else raw * 1024


def _tracer(limit, lines, memory_limit=MEMORY_BUDGET_BYTES,
            block_limit=BLOCK_BUDGET, interval=MEMORY_CHECK_INTERVAL_S,
            deadline=CALL_DEADLINE_S):
    """Trace function enforcing FOUR budgets: events, seconds, live blocks, bytes.

    The DEADLINE is the load-bearing one — see CALL_DEADLINE_S. The other three
    are cheaper and more specific, and each of them has a demonstrated blind
    spot, which is exactly why the deadline backstops all of them.

    Both memory budgets are GROWTH deltas measured from the first sample of this
    call, never absolute ceilings: the absolute figure belongs to whatever else
    the process is doing (in the gym, up to WORKERS sibling threads), and a
    budget that moves with its neighbours is not a budget.

    Two memory signals, because they fail in opposite directions.
    `sys.getallocatedblocks` is free (no syscall), falls again when objects are
    released, and catches the "millions of small objects" runaway; it is blind
    to `bytearray(10**10)`, which is one block. RSS catches exactly that, but is
    a high-water mark that never falls, so alone it would slowly ratchet toward
    a false halt. EITHER budget trips, and both sit far above anything a
    legitimate probe case reaches (whole gym tasks measured at 0.10 GB peak for
    the entire process).

    `interval` is a WALL-CLOCK sampling period, not an event count — see
    MEMORY_CHECK_INTERVAL_S for the measurement that forced that choice.
    Passing `interval=0` samples on every event (tests use it; it is ~25x
    slower).
    """
    count = [0]
    base_rss = [None]
    base_blocks = [None]
    next_at = [0.0]
    started = [None]

    def trace(frame, event, arg):
        count[0] += 1
        if count[0] > limit:
            raise _HardHalt("instruction budget exhausted")
        now = time.monotonic()
        if started[0] is None:
            started[0] = now
        elif deadline is not None and now - started[0] > deadline:
            raise _HardHalt("time budget exhausted")
        if memory_limit is not None or block_limit is not None:
            if now >= next_at[0]:
                next_at[0] = now + interval
                if block_limit is not None:
                    blocks = sys.getallocatedblocks()
                    if base_blocks[0] is None:
                        base_blocks[0] = blocks
                    elif blocks - base_blocks[0] > block_limit:
                        raise _HardHalt("allocation budget exhausted")
                if memory_limit is not None:
                    rss = _rss_bytes()
                    if rss is not None:
                        if base_rss[0] is None:
                            base_rss[0] = rss
                        elif rss - base_rss[0] > memory_limit:
                            raise _HardHalt("memory budget exhausted")
        if frame.f_code.co_filename == SANDBOX_FILENAME:
            if event == "line" or event == "call":
                lines.add(frame.f_lineno)
        return trace
    return trace


def run_case(fn, case):
    """(Outcome, executed line numbers). Never raises."""
    lines = set()
    if fn is None:
        return io.Outcome(ok=False, exc="Unloadable"), frozenset()
    old = sys.gettrace()
    sys.settrace(_tracer(INSTRUCTION_BUDGET, lines))
    try:
        value = fn(*case.call.args, **case.call.as_kwargs())
        return io.Outcome(ok=True, value=value), frozenset(lines)
    except (io.Halt, _HardHalt):
        # ONE outcome for every budget. `_HardHalt` exists only so candidate
        # code cannot swallow the halt (see its docstring); the fingerprint a
        # downstream predicate sees is byte-identical to the catchable form.
        return io.Outcome(ok=False, exc="Halt"), frozenset(lines)
    except BaseException as exc:
        return io.Outcome(ok=False, exc=type(exc).__name__), frozenset(lines)
    finally:
        sys.settrace(old)


def _fingerprint(outcome):
    """Comparable outcome identity. Unrepr-able values fall back to their type,
    which is still a comparison — just a coarser one."""
    if not outcome.ok:
        return ("raised", outcome.exc)
    try:
        text = repr(outcome.value)
    except Exception:
        text = "<unrepr %s>" % type(outcome.value).__name__
    return ("returned", text[:VALUE_REPR_CLIP])


def _execute_matrix(fn_source, name, candidates, span_lines, cases):
    """{'base': [fp...], 'cands': [[fp...]|None], 'cov': [...]}.

    A candidate whose source cannot be built or loaded gets None and takes no
    part in any predicate — it is neither killed nor credited.
    """
    base_fn = _load(fn_source, name)
    base_fps, base_cov = [], []
    for case in cases:
        out, cov = run_case(base_fn, case)
        base_fps.append(_fingerprint(out))
        base_cov.append(cov)

    cand_fps, cand_cov = [], []
    for cand in candidates:
        src = io._apply(span_lines, cand)
        fn = _load(src, name) if src is not None else None
        if fn is None:
            cand_fps.append(None)
            cand_cov.append(None)
            continue
        fps, per_case = [], []
        for case in cases:
            out, c = run_case(fn, case)
            fps.append(_fingerprint(out))
            per_case.append(c)
        cand_fps.append(fps)
        cand_cov.append(per_case)
    return base_fps, base_cov, cand_fps, cand_cov


# ================================================== 4. THE DISCRIMINATION RULE
def _classify_cases(base_fps, cand_fps):
    """Per case: DISCRIMINATING / VACUOUS / INERT (see the header)."""
    useful, vacuous, inert = [], [], []
    live = [f for f in cand_fps if f is not None]
    for i in range(len(base_fps)):
        observed = {f[i] for f in live}
        if not observed:
            continue
        if len(observed) >= 2:
            useful.append(i)
        elif observed == {base_fps[i]}:
            vacuous.append(i)      # passes pre-fix AND post-fix: proves nothing
        else:
            inert.append(i)
    return useful, vacuous, inert


# ======================================================= 5. ORACLE PREDICATES
# Every predicate answers ONE question: given what the function itself already
# says, which candidates behave consistently? A predicate returns a set of
# candidate indices, or None when it does not fire. It may never return an
# empty set — an empty set is a bug, and the caller discards it anyway (rule 3).

_EMPTY_MSG = re.compile(
    r"\bno\s+\w+\s+(provided|supplied|given|found)\b|\bempty\b|\bmissing\b"
    r"|\bat least (one|1)\b|\bis required\b|\bcannot be empty\b|\bno files\b",
    re.I)
_TOOMANY_MSG = re.compile(
    r"\btoo many\b|\bmaximum\b|\bexceed\w*\b|\bat most\b|\btoo large\b"
    r"|\btoo long\b|\blimit\b", re.I)


def _raise_messages(fn):
    """[(message_text, polarity)] for every literal raise message in the body."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        for sub in ast.walk(node.exc):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append(sub.value)
            elif isinstance(sub, ast.JoinedStr):
                out.append("".join(v.value for v in sub.values
                                   if isinstance(v, ast.Constant)
                                   and isinstance(v.value, str)))
    return out


def predicate_raise_message(fn, cases, base_fps, cand_fps):
    """A guard whose message says 'no files provided' must fire on ZERO and stay
    quiet on a valid count. The message is a contract the function wrote about
    itself; a candidate that contradicts it contradicts the function.

    This is the predicate that decides `validate_file_count`, where the bug
    (`if file_count > 1: raise "No files provided"`) raises on the healthy input
    and passes the empty one.
    """
    messages = _raise_messages(fn)
    empty_kind = any(_EMPTY_MSG.search(m) for m in messages)
    many_kind = any(_TOOMANY_MSG.search(m) for m in messages)
    if not empty_kind:
        return None, ""
    # numeric probes only: the contract is about a count
    lows, highs = [], []
    for i, case in enumerate(cases):
        vals = [a for a in case.call.args
                if isinstance(a, int) and not isinstance(a, bool)]
        if not vals:
            continue
        v = vals[0]
        if v <= 0:
            lows.append(i)
        elif 1 <= v <= 2:
            highs.append(i)
    if not lows or not highs:
        return None, ""
    ok = set()
    for ci, fps in enumerate(cand_fps):
        if fps is None:
            continue
        raises_low = all(fps[i][0] == "raised" for i in lows)
        quiet_high = all(fps[i][0] != "raised" for i in highs)
        if raises_low and quiet_high:
            ok.add(ci)
    if not ok:
        return None, ""
    why = "raise-message contract (%s)" % ("empty+limit" if many_kind else "empty")
    return ok, why


# ---------------------------------------------------------- cascade ordering
# THE PREDICATE THAT DECIDES THE CHRONIC SIBLING-LINE SET.
#
# `_map_level`, `_grade_to_band` and their kin are threshold CASCADES: three or
# more `if <same var> OP <const>: return <label>` arms plus a trailing default.
# Such a cascade has a structural invariant no docstring needs to state:
#
#   sweeping the variable monotonically must walk the arms monotonically.
#
# `if g <= 2 / if g > 5 / if g <= 8 / return "9-12"` violates it — g=3..5 falls
# through to arm 3 while g>=6 stops at arm 2, so the arm order inverts mid-sweep
# and the final arm is unreachable. Restoring `g <= 5` restores the walk. This is
# measured by LINE COVERAGE per input, not by reasoning about the operators, so
# it is immune to which operator spelling a candidate happens to use.

MIN_CASCADE_ARMS = 3


def _cascade_arms(fn):
    """Return-lines of an if-cascade over ONE variable, in source order.

    Only a genuine cascade qualifies: >=3 sibling `if`s testing the same left
    operand against a constant/name, each returning. Two arms is a pair of
    independent guards (`validate_file_count`), not a cascade, and this
    predicate deliberately says nothing about those.
    """
    for parent in ast.walk(fn):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        groups = {}
        for stmt in body:
            if not isinstance(stmt, ast.If) or not isinstance(stmt.test, ast.Compare):
                continue
            if len(stmt.test.ops) != 1:
                continue
            rets = [n for n in stmt.body if isinstance(n, ast.Return)]
            if len(stmt.body) != 1 or not rets:
                continue
            # group by the compared OPERAND: `if grade == "K"` and `if g <= 2`
            # are different cascades sharing a body, and only the largest group
            # is a cascade at all.
            groups.setdefault(ast.dump(stmt.test.left), []).append(rets[0].lineno)
        if not groups:
            continue
        arms = max(groups.values(), key=len)
        if len(arms) >= MIN_CASCADE_ARMS:
            tail = [n.lineno for n in body if isinstance(n, ast.Return)
                    and n.lineno > arms[-1]]
            return arms + tail
    return []


def _sweep_value(case):
    """The numeric coordinate of a case, or None when it has no ordered probe."""
    for arg in case.call.args:
        if isinstance(arg, bool):
            continue
        if isinstance(arg, (int, float)):
            return float(arg)
        if isinstance(arg, str):
            try:
                return float(arg)
            except ValueError:
                continue
    return None


def _walks_monotonically(arm_lines, cases, per_case_cov):
    """True when the sweep walks the cascade arms monotonically and reaches >=3.

    None means 'not observable' (too few arms reached) — never False, because an
    unobserved cascade is not a violated one.
    """
    index = {line: i for i, line in enumerate(arm_lines)}
    seen = []
    for ci, case in enumerate(cases):
        x = _sweep_value(case)
        if x is None or per_case_cov is None:
            continue
        hit = [index[l] for l in per_case_cov[ci] if l in index]
        if len(hit) != 1:
            continue
        seen.append((x, hit[0]))
    if len({a for _x, a in seen}) < MIN_CASCADE_ARMS:
        return None
    seen.sort(key=lambda p: p[0])
    arms = [a for _x, a in seen]
    up = all(b >= a for a, b in zip(arms, arms[1:]))
    down = all(b <= a for a, b in zip(arms, arms[1:]))
    return bool(up or down)


def predicate_cascade_order(fn_source, cases, base_cov, cand_cov):
    """Fires only when the ORIGINAL demonstrably violates the cascade walk.

    A function whose cascade is already well-ordered says nothing about which
    candidate is right, so this predicate stays silent there rather than
    inventing a preference.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return None, ""
    arms = _cascade_arms(fn)
    if len(arms) < MIN_CASCADE_ARMS:
        return None, ""
    if _walks_monotonically(arms, cases, base_cov) is not False:
        return None, ""                   # original is fine (or unobservable)
    ok = set()
    for ci, cov in enumerate(cand_cov):
        if cov is None:
            continue
        if _walks_monotonically(arms, cases, cov) is True:
            ok.add(ci)
    if not ok:
        return None, ""
    return ok, "restores the threshold cascade's monotone walk (%d arms)" % len(arms)


def predicate_branch_preservation(fn_source, base_cov, cand_cov):
    """A fix may revive dead code; it may not KILL live code.

    A candidate that renders a branch body unreachable which the original
    reaches has broken something that worked — the classic cost of mutating a
    correct sibling line. Eliminating those is safe in the direction that
    matters: the gold fix restores intended behaviour, it does not delete a
    working arm.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return None, ""
    bodies = {n.lineno for n in ast.walk(fn)
              if isinstance(n, (ast.Return, ast.Raise))}
    if len(bodies) < 3:
        return None, ""
    live = set().union(*base_cov) & bodies if base_cov else set()
    if len(live) < 2:
        return None, ""
    ok = set()
    for ci, cov in enumerate(cand_cov):
        if cov is None:
            continue
        reached = set().union(*cov) & bodies if cov else set()
        if live <= reached:
            ok.add(ci)
    if not ok or len(ok) == len([c for c in cand_cov if c is not None]):
        return None, ""
    return ok, "keeps every branch the original still reaches live"


def predicate_docstring(fn_source, span_lines, candidates):
    """Reuse the intent oracle: the docstring's own stated contract, executed.

    Delegated rather than reimplemented — it is already proven to kill zero gold
    candidates over this bench, and duplicating its miners would duplicate its
    bugs.
    """
    try:
        res = io.discriminate(candidates, fn_source, span_lines)
    except Exception:
        return None, ""
    if not res.get("applied") or not res.get("killed"):
        return None, ""
    killed = {id(c) for c, _e, _w in res["killed"]}
    ok = {ci for ci, c in enumerate(candidates) if id(c) not in killed}
    if not ok or len(ok) == len(candidates):
        return None, ""
    return ok, "docstring contract (%d expectation kill(s))" % len(res["killed"])


# MEASURED AND DELETED — two predicates that looked obviously right and were not.
#
#   crash_freedom     ("a fix must stop the None crash")  killed the gold on
#                     9 of the 10 tasks where it fired.
#   behaviour_change  ("a fix must change behaviour")     killed the gold on
#                     11 of the 17 tasks where it fired.
#
# Both failed for the same reason: the synthesized inputs frequently never REACH
# the mutated line (a `db`/session parameter makes the body raise long before
# it), so the gold candidate looks like a no-op and a sibling candidate that
# happens to sit on a reachable early line looks like the fix. Behavioural
# evidence is only evidence WHERE THE LINE EXECUTES. The surviving form of that
# idea is `predicate_no_op` below, which is gated on exactly that and therefore
# only ever removes candidates proven inert on their own executed line.


def predicate_no_op(fn_source, span_lines, candidates, cases, base_fps,
                    cand_fps, base_cov, cand_cov):
    """Removes candidates PROVABLY equivalent to the line they replace.

    The proof is at the EXPRESSION level — witness_replay's truth table over the
    boundary witness rows — not at the whole-function level. That distinction is
    the whole safety argument:

      An execution-level version of this predicate ("the line ran and nothing
      moved") killed the gold on `_build_equivalent_forms` and `wrap_untrusted`,
      because a line can execute while the synthesized inputs never reach the
      VALUES that separate the two operators. Inert-under-my-probes is a fact
      about my probes. Equal-truth-table is a fact about the code.

    Unsupported expressions abstain (they survive), per the fail-open law.
    """
    ok = set()
    dropped = 0
    for ci, fps in enumerate(cand_fps):
        if fps is None:
            continue
        try:
            _off, old_line, new_line = candidates[ci]
            old_node, _ = wr.extract_condition(old_line)
            new_node, _ = wr.extract_condition(new_line)
            if old_node is None or new_node is None:
                ok.add(ci)
                continue
            rows = wr.witness_rows(wr.free_variables(old_node, new_node))
            if wr.truth_table(old_node, rows) == wr.truth_table(new_node, rows):
                dropped += 1              # same behaviour on every witness row
                continue
        except Exception:
            pass                          # unsupported subset: it SURVIVES
        ok.add(ci)
    if not dropped or not ok:
        return None, ""
    return ok, "drops %d candidate(s) with a truth table identical to the original" % dropped


# ------------------------------------------------- self-describing boundaries
# The last mile on the chronic set is INCLUSIVITY: `>= INFO` vs `> INFO`,
# `<= 5` vs `< 5`, `< 18` vs `<= 18`. No behavioural predicate separates those —
# both spellings reverse the direction, both keep the cascade monotone. But the
# code frequently NAMES its own boundary and then the answer is checkable:
#
#   `if levelno >= logging.INFO: return "info"`   the arm returns the constant's
#                                                 own name, so it must fire AT it
#   `if g <= 5: return "3-5"`                     the label states its interval,
#                                                 so grade 5 must land in it
#   `"...for learners 18+"`                       the docstring states which side
#                                                 of 18 the boundary belongs to
#
# Each is a claim the function makes about itself, verified by running it.

_RANGE_LABEL = re.compile(r"^([A-Za-z]|\d+)-(\d+)$")
_DOC_BOUNDARY = re.compile(
    r"(\d+)\s*\+|(?:at least|older than or equal to|minimum of)\s*(\d+)"
    r"|(\d+)\s*(?:or older|and above|and over|or more)", re.I)


def _labelled_arms(fn_source):
    """[(compared value, label, rhs name)] for arms that describe themselves."""
    fn = _fn_node(fn_source)
    if fn is None:
        return []
    ns = _namespace(fn_source)
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if len(node.test.ops) != 1 or len(node.body) != 1:
            continue
        ret = node.body[0]
        if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Constant):
            continue
        label = ret.value.value
        if not isinstance(label, str):
            continue
        rhs = node.test.comparators[0]
        rhs_name = rhs.attr if isinstance(rhs, ast.Attribute) else (
            rhs.id if isinstance(rhs, ast.Name) else "")
        try:
            value = eval(compile(ast.Expression(rhs), SANDBOX_FILENAME, "eval"),
                         dict(ns))                              # noqa: S307
        except Exception:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append((float(value), label, rhs_name))
    return out


def _required_points(fn_source):
    """[(sweep value, required returned label)] the function claims about itself."""
    reqs = []
    for value, label, rhs_name in _labelled_arms(fn_source):
        if rhs_name and label.strip().lower() == rhs_name.strip().lower():
            reqs.append((value, label))          # arm named after its constant
        m = _RANGE_LABEL.match(label.strip())
        if m:
            lo, hi = m.group(1), m.group(2)
            reqs.append((float(hi), label))
            if lo.isdigit():
                reqs.append((float(lo), label))
    seen, out = set(), []
    for point in reqs:
        if point not in seen:
            seen.add(point)
            out.append(point)
    return out


def predicate_self_describing_label(fn_source, cases, base_fps, cand_fps):
    """A returned label that names its own boundary must actually cover it.

    Fires only where the ORIGINAL breaks its own claim — a function already
    keeping its word says nothing about which candidate is right.
    """
    reqs = _required_points(fn_source)
    if not reqs:
        return None, ""
    index = {}
    for i, case in enumerate(cases):
        x = _sweep_value(case)
        if x is not None:
            index.setdefault(x, i)
    checks = [(index[v], label) for v, label in reqs if v in index]
    if not checks:
        return None, ""

    def honours(fps):
        return all(fps[i] == ("returned", repr(label)) for i, label in checks)

    if honours(base_fps):
        return None, ""
    ok = {ci for ci, fps in enumerate(cand_fps) if fps is not None and honours(fps)}
    if not ok:
        return None, ""
    return ok, ("returns the label that names the boundary at %d checked point(s)"
                % len(checks))


def predicate_docstring_boundary(fn_source, cases, base_fps, cand_fps):
    """`"...for learners 18+"` says 18 belongs with 19, not with 17.

    A pure inclusivity statement, which is exactly the axis every behavioural
    predicate is blind to. Checked by comparing the outputs at N-1, N and N+1.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return None, ""
    doc = ast.get_docstring(fn) or ""
    match = _DOC_BOUNDARY.search(doc)
    if not match:
        return None, ""
    n = float(next(g for g in match.groups() if g))
    index = {}
    for i, case in enumerate(cases):
        x = _sweep_value(case)
        if x is not None:
            index.setdefault(x, i)
    if not all(v in index for v in (n - 1, n, n + 1)):
        return None, ""
    below, at, above = index[n - 1], index[n], index[n + 1]

    def honours(fps):
        return fps[at] == fps[above] and fps[at] != fps[below]

    if honours(base_fps):
        return None, ""
    ok = {ci for ci, fps in enumerate(cand_fps) if fps is not None and honours(fps)}
    if not ok:
        return None, ""
    return ok, "puts %g on the inclusive side the docstring names" % n


# ------------------------------------------------------ sibling disagreement
# Functions repeat themselves, and the repetition is a free oracle. Both of the
# remaining chronic tasks carry a second copy of the very comparison that was
# flipped:
#
#   _update_lesson_progress   `lesson_progress.passed = ...score < 70`
#                             `LessonProgress(..., passed=(score >= 70), ...)`
#   _build_equivalent_forms   `if abs(numer / denom - val) > 1e-9:`
#                             `if abs(val - int_val)     < 1e-9:`
#
# One flipped copy makes the function contradict itself. Restoring agreement is
# the fix; every other candidate leaves the contradiction standing. The rule
# fires ONLY when the original is self-contradictory, so a function that was
# always consistent is never second-guessed.

def _compare_signature(node, target_name=""):
    """(what is being compared, the literal it is compared to), or None.

    Deliberately coarse: the LEFT side is reduced to a name — the assigned
    target, or the callee of a call like `abs(...)` — so two copies of one idea
    written over different operands still match.
    """
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    rhs = node.comparators[0]
    if not isinstance(rhs, ast.Constant) or isinstance(rhs.value, (str, bool)):
        return None
    left = node.left
    if target_name:
        key = target_name
    elif isinstance(left, ast.Call) and isinstance(left.func, ast.Name):
        key = left.func.id
    elif isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute):
        key = left.func.attr
    else:
        return None
    return (key, repr(rhs.value))


def _sibling_groups(fn):
    """{signature: {op name: [line numbers]}} over the whole function body."""
    groups = {}

    def record(node, target=""):
        sig = _compare_signature(node, target)
        if sig is None:
            return
        op = type(node.ops[0]).__name__
        groups.setdefault(sig, {}).setdefault(op, []).append(node.lineno)

    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Compare):
            for tgt in node.targets:
                name = (tgt.attr if isinstance(tgt, ast.Attribute)
                        else tgt.id if isinstance(tgt, ast.Name) else "")
                if name:
                    record(node.value, name)
        elif isinstance(node, ast.keyword) and isinstance(node.value, ast.Compare) \
                and node.arg:
            record(node.value, node.arg)
        elif isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Compare):
            record(node.test)
    return groups


def predicate_role_consistency(fn_source, span_lines, candidates):
    """Candidates that use every name in a ROLE the original function already gives it.

    FINDINGS E247. Measured: the ten runtime predicates leave 91% of oracle abstentions at
    "no predicate fired" while the candidate menus (34-85 mutants of one function) are full
    of NAME SWAPS drawn from the function's own identifiers:

        db.query(Learner)        ->  learner_id.query(Learner)     receiver swapped
        raise HTTPException(...) ->  raise db(...)                 callee swapped

    These are syntactically perfect, so `compile` passes them, and semantically impossible,
    so no runtime case is needed to reject them -- the ORIGINAL SOURCE is the evidence. A
    name that is never a call receiver in the real function has no business being one in a
    repair of it.

    Sized before it was written, over 1,046 candidates on 20 bench tasks:
        bad_receiver  6.7% of candidates eliminated, 0 gold kills
        bad_callee    3.7%                            0 gold kills
    versus the entire runtime battery's measured 6%. No model call.

    FAIL-OPEN in three places, deliberately, because a wrong elimination makes a task
    unfixable forever: an unparseable original abstains, an original with no observed
    receivers/callees abstains rather than flagging everything, and a candidate we cannot
    parse is KEPT rather than killed. Returns (keep_indices, why) or (None, "") per the
    ladder contract; the ladder additionally refuses any narrowing that empties the set.
    """
    try:
        tree = ast.parse(fn_source)
    except SyntaxError:
        return None, ""

    receivers, callees, known = set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            known.add(n.id)                  # every name the ORIGINAL function mentions
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                receivers.add(n.func.value.id)
            elif isinstance(n.func, ast.Name):
                callees.add(n.func.id)
    if not receivers and not callees:
        return None, ""                      # no role evidence: abstain, never flag all

    keep, flagged = set(), 0
    for ci, cand in enumerate(candidates):
        try:
            line = str(cand[2])
            node = ast.parse(line.strip()).body[0]
        except Exception:
            keep.add(ci)                     # unparseable -> KEEP (fail open)
            continue
        bad = False
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            # ONLY FLAG A NAME THE ORIGINAL ALREADY KNOWS (FINDINGS E251, adversarial
            # test). Flagging ANY unprecedented role would kill the true fix on 8 of 21 real
            # bugs, because real repairs legitimately introduce new names — `getattr`,
            # `isinstance`, `SessionLocal`, `sorted`, a new helper. The earlier "0 gold kills
            # over 1,046 candidates" could never have caught that: those candidates are
            # NAME-SWAP MUTANTS of the parent and by construction almost never introduce a
            # new name, so the population could not falsify the rule.
            #
            # The real defect is narrower and this is the whole rule: a name the function
            # ALREADY USES, appearing in a role it NEVER has (`learner_id.query(...)` where
            # `learner_id` is an id and `db` is the session). A brand-new name is a helper or
            # an import and is none of our business.
            if (receivers and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in known
                    and n.func.value.id not in receivers):
                bad = True
                break
            if (callees and isinstance(n.func, ast.Name)
                    and n.func.id in known
                    and n.func.id not in callees):
                bad = True
                break
        if bad:
            flagged += 1
        else:
            keep.add(ci)

    if not flagged:
        return None, ""                      # nothing to say about this menu
    return keep, ("%d candidate(s) use a name in a role the original function never gives it"
                  % flagged)


def predicate_sibling_agreement(fn_source, span_lines, candidates):
    """Candidates that make the function agree with its own second copy."""
    fn = _fn_node(fn_source)
    if fn is None:
        return None, ""
    split = {sig: ops for sig, ops in _sibling_groups(fn).items() if len(ops) > 1}
    if not split:
        return None, ""                   # the function does not contradict itself
    ok, decided = set(), 0
    for ci, cand in enumerate(candidates):
        try:
            _off, old_line, new_line = cand
            patched = io._apply(span_lines, cand)
            if patched is None:
                ok.add(ci)
                continue
            new_fn = _fn_node(patched)
            if new_fn is None:
                ok.add(ci)
                continue
            groups = _sibling_groups(new_fn)
            relevant = [sig for sig in split if sig in groups]
            if not relevant:
                continue                  # the conflict is untouched: no credit
            if all(len(groups[sig]) == 1 for sig in relevant):
                decided += 1
                ok.add(ci)                # this candidate resolves the conflict
        except Exception:
            ok.add(ci)                    # unreadable candidate: abstain
    if not decided or not ok or len(ok) == len(candidates):
        return None, ""
    return ok, ("makes the function agree with its own duplicate comparison "
                "(%d signature(s))" % len(split))


def _positive_flag_ops(fn):
    """[(name, op)] for every `<positive predicate> = <value> <op> <number>`."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.keyword)):
            continue
        value = node.value
        if not isinstance(value, ast.Compare) or len(value.ops) != 1:
            continue
        rhs = value.comparators[0]
        if not isinstance(rhs, ast.Constant) or isinstance(rhs.value, (str, bool)):
            continue
        if isinstance(node, ast.keyword):
            names = [node.arg or ""]
        else:
            names = [t.attr if isinstance(t, ast.Attribute)
                     else t.id if isinstance(t, ast.Name) else ""
                     for t in node.targets]
        for name in names:
            if name and wr._POSITIVE_PREDICATE.match(name):
                out.append((name, type(value.ops[0])))
    return out


def predicate_positive_flag(fn_source, span_lines, candidates):
    """`passed = score < 70` is backwards on its face: a positively-named flag
    reads TRUE above its threshold, not below it.

    The naming convention is witness_replay's `_POSITIVE_PREDICATE`, reused
    rather than restated. Fires only when the original violates it, and judges
    only candidates that leave every such flag reading the conventional way.
    """
    fn = _fn_node(fn_source)
    if fn is None:
        return None, ""
    flags = _positive_flag_ops(fn)
    if not flags or all(op in wr._GREATER_OPS for _n, op in flags):
        return None, ""                   # nothing stated, or nothing violated
    ok = set()
    for ci, cand in enumerate(candidates):
        try:
            patched = io._apply(span_lines, cand)
            new_fn = _fn_node(patched) if patched else None
            if new_fn is None:
                ok.add(ci)
                continue
            if all(op in wr._GREATER_OPS for _n, op in _positive_flag_ops(new_fn)):
                ok.add(ci)
        except Exception:
            ok.add(ci)
    if not ok or len(ok) == len(candidates):
        return None, ""
    return ok, "leaves every positively-named flag reading TRUE above its threshold"


def predicate_symptom_direction(symptom_class, candidates, surviving):
    """THE LAST RUNG: the symptom's own words, applied to what is still standing.

    A `cmp_flip` symptom says the comparison runs the WRONG DIRECTION, so a
    candidate that reverses it is consistent with the symptom and one that
    rewrites an operand is not. Direction detection is witness_replay's
    `reverses_direction`, called rather than reimplemented.

    MEASURED AND REMOVED: a second rung preferring the STRICTNESS-PRESERVING
    reversal (`> 5` -> `< 5` over `<= 5`). It reads as obviously right and it is
    wrong on this bench: the gold flips strictness on `_grade_to_band`
    (`g > 5` -> `g <= 5`, which is what keeps the band partition complete) and on
    `_update_lesson_progress` (`score < 70` -> `score >= 70`). It killed the gold
    on both while rescuing one task. Inclusivity is decided by what the function
    says it means, never by the symptom label — so where two reversals differ
    only in the boundary, the honest verdict is AMBIGUOUS.
    """
    if symptom_class != "cmp_flip" or len(surviving) < 2:
        return None, ""
    reversing = set()
    for ci in surviving:
        try:
            _off, old_line, new_line = candidates[ci]
            old_node, _ = wr.extract_condition(old_line)
            new_node, _ = wr.extract_condition(new_line)
            if old_node is None or new_node is None:
                continue
            if wr.reverses_direction(old_node, new_node):
                reversing.add(ci)
        except Exception:
            continue
    if not reversing:
        return None, ""
    return reversing, "reverses the comparison direction the symptom names"


# ================================================================ 6. VERDICT
def synthesize_discriminating_test(fn_source, candidates, symptom_class=None,
                                   span_lines=None):
    """THE ENTRY POINT.

    fn_source   : the function's source (the span, as the fixer holds it)
    candidates  : the mutation menu, [(abs_line_idx, old_line, new_line), ...]
    symptom_class: the task's symptom family, when the harness knows it
    span_lines  : the span as a line list; defaults to fn_source.splitlines()

    Returns a DiscriminationResult. Fail-open at every step: any failure is
    NO_SIGNAL with the reason attached, never an elimination.
    """
    result = DiscriminationResult()
    try:
        return _synthesize(fn_source, candidates, symptom_class, span_lines,
                           result)
    except Exception as exc:                      # law 1: rather learn nothing
        result.verdict = NO_SIGNAL
        result.reason = "engine error: %s" % type(exc).__name__
        result.survivors = list(candidates or [])
        return result


def _synthesize(fn_source, candidates, symptom_class, span_lines, result):
    cands = list(candidates or [])
    result.survivors = list(cands)
    if len(cands) < 2:
        result.reason = "nothing to discriminate between"
        return result
    span = list(span_lines) if span_lines else (fn_source or "").splitlines()
    fn = _fn_node(fn_source)
    if fn is None:
        result.reason = "function did not parse"
        return result

    budget = max(MIN_CASES, MAX_EXECUTIONS // max(1, len(cands)))
    cases = synthesize_cases(fn_source, cands, symptom_class,
                             max_cases=min(MAX_CASES, budget))
    result.cases = cases
    if not cases:
        result.reason = "no case template applied"
        return result

    base_fps, base_cov, cand_fps, cand_cov = _execute_matrix(
        fn_source, fn.name, cands, span, cases)
    live = [ci for ci, f in enumerate(cand_fps) if f is not None]
    if len(live) < 2:
        result.reason = "fewer than two candidates are executable"
        return result

    useful, vacuous, inert = _classify_cases(base_fps, cand_fps)
    result.discriminating, result.vacuous, result.inert = useful, vacuous, inert
    if not useful:
        result.reason = ("no discriminating case: %d vacuous, %d inert"
                         % (len(vacuous), len(inert)))
        return result

    # ---- the predicate ladder, strongest first. Each narrows; none may empty.
    surviving = set(live)
    fired = []
    ladder = [
        ("raise_message",
         predicate_raise_message(fn, cases, base_fps, cand_fps)),
        ("docstring",
         predicate_docstring(fn_source, span, cands)),
        ("cascade_order",
         predicate_cascade_order(fn_source, cases, base_cov, cand_cov)),
        ("branch_preservation",
         predicate_branch_preservation(fn_source, base_cov, cand_cov)),
        ("self_describing_label",
         predicate_self_describing_label(fn_source, cases, base_fps, cand_fps)),
        ("docstring_boundary",
         predicate_docstring_boundary(fn_source, cases, base_fps, cand_fps)),
        ("sibling_agreement",
         predicate_sibling_agreement(fn_source, span, cands)),
        # ROLE CONSISTENCY IS DELIBERATELY NOT IN THIS LADDER. See FINDINGS E251.
        #
        # It measured well — 82 of 704 survivors removed (11.6%), 0 gold kills over 1,046
        # candidates, and it moved the oracle from speaking on 45% of tasks to 80%. Then
        # adversarial review pointed out the measurement was close to circular: our
        # candidates are NAME-SWAP MUTANTS of the parent, and the rule detects name swaps, so
        # that population could never falsify it.
        #
        # Tested properly against the REAL gold patches (probes/role_rule_adversarial.py):
        #     original form  -> would eliminate the true fix on 8 of 21 real bugs (38%)
        #     narrowed form  -> 1 of 21 (5%)   [`normalize_generated_content`]
        # A wrong elimination makes a task permanently unfixable, and our own standing rule is
        # that a single gold kill is a VETO rather than a tuning opportunity. 5% is not zero,
        # so it does not get to narrow a survivor set.
        #
        # The predicate stays built, tested and callable for use as a RANKING signal (which
        # cannot make a task unfixable). Re-admitting it as an eliminator requires restricting
        # rejection to roles that are PROVABLY impossible — a name bound to a non-callable in
        # scope — not merely unprecedented, and a fresh adversarial run showing 0/21.
        ("positive_flag",
         predicate_positive_flag(fn_source, span, cands)),
        ("no_op",
         predicate_no_op(fn_source, span, cands, cases, base_fps, cand_fps,
                         base_cov, cand_cov)),
    ]
    for name, (ok, why) in ladder:
        if ok is None:
            continue
        narrowed = surviving & set(ok)
        if not narrowed:
            continue                      # RULE 3: fail open, never empty
        fired.append((name, why, len(narrowed)))
        surviving = narrowed

    # the symptom rung reads the CURRENT survivor set, so it runs after the loop
    ok, why = predicate_symptom_direction(symptom_class, cands, surviving)
    if ok is not None:
        narrowed = surviving & set(ok)
        if narrowed:
            fired.append(("symptom_direction", why, len(narrowed)))
            surviving = narrowed

    result.predicates = fired
    ordered = [c for ci, c in enumerate(cands) if ci in surviving]
    result.survivors = ordered

    if not fired:
        result.verdict = NO_SIGNAL
        result.reason = "no oracle predicate fired"
        return result
    if len(ordered) == 1:
        winner_idx = next(iter(surviving))
        result.verdict = SELECTED
        result.selected = ordered[0]
        result.cases_cited = _cite(cases, useful, winner_idx, cand_fps, base_fps)
        result.reason = "; ".join("%s: %s" % (n, w) for n, w, _k in fired)
        return result
    result.verdict = AMBIGUOUS
    result.reason = ("%d candidates remain after %s"
                     % (len(ordered), ", ".join(n for n, _w, _k in fired)))
    return result


def _cite(cases, useful, winner_idx, cand_fps, base_fps, limit=4):
    """The cases that actually back the pick: discriminating cases where the
    winner differs from the original. A citation nobody can check is decoration.
    """
    out = []
    fps = cand_fps[winner_idx]
    for i in useful:
        if fps is None:
            break
        if fps[i] != base_fps[i]:
            out.append("%s -> %s (was %s)"
                       % (cases[i].label, fps[i][1][:40], base_fps[i][1][:40]))
        if len(out) >= limit:
            break
    return out
