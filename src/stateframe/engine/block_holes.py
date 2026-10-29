"""BOUNDED BLOCK-HOLE GENERATION — the harness owns the position, the model
writes the statements.

WHY THIS LANE EXISTS (measured, not assumed)
--------------------------------------------
`probes/block_miner.py reach` reports 0/7 with a PASSING planted control: the
block-TEMPLATE family's structure is sound and its corpus simply does not carry
the golds. Domain fix blocks do not repeat across repositories often enough for
retrieval to pay, so the donor is the wrong instrument for this board.

The second measurement that moved this lane to primary: of the 21 v3 golds only
7 are single-block ADDITIONS. Fourteen are not — most of them add
statements AND delete or rewrite neighbouring ones. A block TEMPLATE cannot
express that (one template is one contiguous inserted run — see
`blocks.BLOCK_PUNTS`), but the EditSet
substrate can: a replacement of a contiguous statement run by new text is the
same single atomic splice as an insertion. So this family enumerates BOTH.

THE DIVISION OF LABOUR
----------------------
    code   chooses WHERE      (`enumerate_sites`)      - never the model
    code   chooses the RULES  (`build_envelope`)       - never the model
    model  writes the BLOCK   (one no-think call)      - nothing else
    code   VALIDATES          (`validate_block`)       - authority, fail-closed
    code   APPLIES atomically (`make_block_edit`)      - one `EditSet`

The model never sees gold, never sees the tests, never chooses a position, and
cannot emit a patch because the protocol gives it nowhere to put one. Every
survivor becomes an ordinary `edit_sets.EditSet` and enters the FULL pipeline —
there is no bypass, exactly as with the typed holes in `holes.py`.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
`probes/block_hole_reachability.py` injects the GOLD block AS IF THE MODEL HAD
AUTHORED IT and asks only: does position enumeration find the spot, does the
envelope admit that text, and does the EditSet express the change? That is the
PROTOCOL CEILING and nothing else. Whether a 1B can write the block is a LIVE
experiment this lane does not run and must never claim (see `BLOCK_HOLE_PUNTS`
and `BLOCK_HOLE_ENABLED`).
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import re
import textwrap

import blocks as B
import edit_sets as es

_BUILTIN_NAMES = frozenset(dir(builtins))

SCHEMA_VERSION = 1

# ------------------------------------------------------------------ budgets
#: An envelope is a CEILING, not a target. `MAX_STATEMENTS` matches `blocks`
#: deliberately: two families that disagree about how many STATEMENTS make a
#: block would make the punt classes in the two reachability tables
#: incomparable. It is also the bound that does the real work below.
MAX_STATEMENTS = B.MAX_BLOCK_STATEMENTS       # 6

#: Raised from `blocks.MAX_BLOCK_AST_NODES` (96) on measured evidence:
#: `update_experiment`'s gold is TWO statements over sixteen lines and counts
#: 100 nodes — four over. Its density is entirely LITERAL COLLECTION ELEMENTS
#: (a six-name set and a six-key dict), not control flow, which is precisely
#: the kind of block a node cap should not be the thing rejecting.
#:
#: 112 is derived, not picked: `MAX_LINES` (20) at the ~6 nodes/line that dense
#: literal Python actually carries is the most a legal block could hold anyway,
#: so this cap now sits just under the line cap's own implied ceiling rather
#: than cutting across it. `MAX_STATEMENTS` (6) is untouched and remains the
#: real complexity bound. The planted `envelope_nodes` control proves a
#: node-dense invalid block is still rejected after the raise.
MAX_AST_NODES = 112

#: The ONE budget where this family deliberately diverges from
#: `blocks.MAX_BLOCK_LINES` (15), raised on MEASURED evidence not on taste.
#: Four v3 golds sat just over the old cap and nothing else was wrong with them
#:
#:     update_experiment   insert   16 raw lines   (cap was 15)
#:     _runner_ws_uuid     insert   17 raw lines   (cap was 15)
#:     _env_guidance       replace  span + 17      (cap was span + 15)
#:     _compose_test       replace  span + 17      (cap was span + 15)
#:
#: 20 covers all four with three lines of headroom and no more. It is NOT a
#: licence to write longer code: `MAX_STATEMENTS` (6) and `MAX_AST_NODES` (96)
#: are untouched, so a 20-line block is still at most six statements. The extra
#: lines buy WRAPPED calls and multi-line string literals, which is exactly
#: what those four golds are, and nothing else. The planted `lines` control in
#: `probes/block_hole_reachability.py` proves a block one line past the raised
#: cap is still rejected, and rejected BY THE LINE RULE specifically.
#:
#: CAVEAT, stated because the comment above promises comparability: the
#: `envelope_lines` punt class is now NOT comparable across the two families.
#: Every other punt class still is.
MAX_LINES = 20

#: How many parent statements one replacement site may cover. A wider run is
#: not "a block hole" any more, it is a rewrite of the function, and the model
#: would be authoring behaviour nobody bounded.
#:
#: Raised 4 -> 6 on measured evidence: `add_task_update` is one contiguous run
#: of SIX statements in a single list (three interleaved edits the family
#: replaces as one splice) and was the only v3 gold in reach of a raise. The
#: bound is now the SAME number as `MAX_STATEMENTS`, which is the principled
#: stopping point: a block may replace at most as many statements as a block is
#: allowed to CONTAIN. Anything wider is a rewrite by another name.
#:
#: Deliberately NOT raised far enough for `runner_schedule_reinit`, whose gold
#: spans TWENTY statements across ~100 lines — see
#: `BLOCK_HOLE_PUNTS["replacement_span_is_bounded"]`. That row is an honest
#: permanent punt, not a coverage gap waiting for a bigger number.
MAX_REPLACE_STATEMENTS = 6

#: Sites offered to the model per span. The enumerator is exhaustive and
#: deterministic; this is the RUNTIME budget a fixer spends, and the probe
#: lifts it (`limit=None`) exactly like `blocks.TargetFacts(point_limit=None)`.
MAX_SITES = 8

#: Samples per (site, envelope). A failed sample is a failed sample: it is
#: counted and discarded, NEVER retried into existence (`holes.sample_fills`).
BLOCK_SAMPLES = 3

#: Answer tokens for the authoring call. Not a server `max_tokens` — the seam
#: adds the rule #11d reasoning headroom on top.
ANSWER_TOKENS = 320

#: OFF until the live referendum deliberately flips it. A family that is on by
#: default before anyone measured its live fill quality is a claim, not a
#: result.
BLOCK_HOLE_ENABLED = False

BLOCK_HOLE_PUNTS = {
    "no_live_fill_measurement":
        "This module measures the PROTOCOL only. Whether a 1B writes a "
        "correct block is a live experiment; nothing here may be quoted as "
        "fill quality.",
    "banned_imports_are_permanent":
        "A block hole may never emit an import, def, class, global/nonlocal "
        "or exec. FIVE v3 golds add a function-local import and are therefore "
        "OUT OF SCOPE by construction, not by corpus coverage. (Four while "
        "the line cap was 15: `_runner_ws_uuid` read as `envelope_lines` "
        "because the size gate runs first, and the raise to 20 only revealed "
        "the permanent blocker underneath. That row was never reachable.)",
    "single_contiguous_run":
        "One site is one contiguous statement run. A gold that edits two "
        "non-adjacent places is a composite and is reported as such.",
    "no_semantic_effect_check":
        "The envelope constrains names, calls, size and syntax. It does not "
        "prove the block's EFFECT; that is the witness's job downstream.",
    "no_attribute_typing":
        "A call on an IN-SCOPE receiver is admitted whatever the attribute "
        "name: this family has no type for a parameter, and rejecting unknown "
        "methods would throw away every legitimate block. An unknown RECEIVER "
        "is still rejected.",
    "must_bind_is_syntactic":
        "`must_bind` requires the replacement to BIND the names the removed "
        "run bound and later statements read. It does not check the value.",
    "replacement_span_is_bounded":
        "A replacement site covers at most MAX_REPLACE_STATEMENTS (6) parent "
        "statements — the same number a block may contain. One v3 gold "
        "(runner_schedule_reinit) rewrites TWENTY statements across ~100 "
        "lines; that is a function rewrite, not a block, and it is a "
        "PERMANENT punt rather than a cap waiting to be raised.",
}

# ------------------------------------------------------------------- roles
#: Same vocabulary as `blocks.ROLES` so the two families' telemetry joins.
ROLES = B.ROLES
ROLE_VERBS = {
    "guard": "Reject or exit early when the unsafe condition holds.",
    "sanitize": "Normalize the value before it is used.",
    "revalidate": "Re-check the value after the barrier and stop if invalid.",
    "loop_guard": "Skip or stop this iteration when the condition holds.",
    "other": "Write the statements the description calls for.",
}

_ROLE_WORDS = (
    # Ordered: the FIRST match wins, so the vocabulary is a decision table and
    # not a bag of overlapping keywords whose outcome depends on dict order.
    ("loop_guard", ("each iteration", "per item", "loop", "iterate",
                    "iteration", "skip the item")),
    ("revalidate", ("revalidate", "re-validate", "recheck", "re-check",
                    "after the", "stale", "reload", "refresh")),
    ("sanitize", ("sanitize", "sanitise", "normalize", "normalise", "strip",
                  "escape", "coerce", "clean")),
    ("guard", ("guard", "missing", "none", "null", "empty", "blank", "invalid",
               "unsafe", "reject", "before", "crash", "traceback", "raises",
               "attributeerror", "keyerror", "typeerror", "valueerror")),
)


def classify_role(symptom):
    """Role verb inferred from the SYMPTOM text, deterministically.

    Purely a prompt-shaping hint. It never gates a candidate: an envelope is
    the same set of hard rules whatever the role says, so a wrong guess costs
    one sentence of prompt and never admits an illegal block.
    """
    text = (symptom or "").lower()
    for role, words in _ROLE_WORDS:
        if any(w in text for w in words):
            return role
    return "other"


# -------------------------------------------------------- banned constructs
#: The model may not change the module's shape, its imports, or its binding
#: scope, and may not reach the dynamic-execution builtins. Every entry is a
#: SEPARATE reason code so a rejection names the rule it broke.
REJECT_BANNED_IMPORT = "REJECT_BANNED_IMPORT"
REJECT_BANNED_DEF = "REJECT_BANNED_DEF"
REJECT_BANNED_CLASS = "REJECT_BANNED_CLASS"
REJECT_BANNED_GLOBAL = "REJECT_BANNED_GLOBAL"
REJECT_BANNED_LAMBDA = "REJECT_BANNED_LAMBDA"

BANNED_NODE_REASONS = (
    ((ast.Import, ast.ImportFrom), REJECT_BANNED_IMPORT),
    ((ast.FunctionDef, ast.AsyncFunctionDef), REJECT_BANNED_DEF),
    ((ast.ClassDef,), REJECT_BANNED_CLASS),
    ((ast.Global, ast.Nonlocal), REJECT_BANNED_GLOBAL),
    ((ast.Lambda,), REJECT_BANNED_LAMBDA),
)

BANNED_CALLS = frozenset(("exec", "eval", "compile", "__import__", "globals",
                          "locals", "vars", "setattr", "delattr"))

REJECT_EMPTY = "REJECT_EMPTY"
REJECT_PARSE = "REJECT_PARSE"
REJECT_LINES = "REJECT_LINES"
REJECT_STATEMENTS = "REJECT_STATEMENTS"
REJECT_NODES = "REJECT_NODES"
REJECT_UNKNOWN_NAME = "REJECT_UNKNOWN_NAME"
REJECT_UNKNOWN_CALL = "REJECT_UNKNOWN_CALL"
REJECT_MUST_BIND = "REJECT_MUST_BIND"
REJECT_INDENT = "REJECT_INDENT"
REJECT_PLACEHOLDER = "REJECT_PLACEHOLDER"
REJECT_PROTOCOL = "REJECT_PROTOCOL"

#: Soft: reported on the verdict, never a rejection. A guard that is correct in
#: every way except that it wrote `'x'` where the file writes `"x"` is not a
#: candidate worth throwing away, and a validator that says otherwise is
#: optimising for cosmetics against measured value.
STYLE_QUOTE = "style_quote"
STYLE_LINE_LENGTH = "style_line_length"

MAX_STYLE_LINE = 79


def _digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# =========================================================== 1. THE SITES
class Site:
    """One legal place a block may be written, chosen by CODE.

    ``kind == "insert"``  the block goes BEFORE ``line``; nothing is removed.
    ``kind == "replace"`` the block REPLACES span lines ``line..end_line``.

    Both are one contiguous splice, which is what makes either one atomic.
    """

    __slots__ = ("kind", "line", "end_line", "indent", "list_role",
                 "loop_depth", "try_depth", "in_handler", "before_kind",
                 "after_kind", "n_stmts", "subjects", "index")

    def __init__(self, kind, line, end_line, indent, list_role, loop_depth,
                 try_depth, in_handler, before_kind, after_kind, n_stmts=0,
                 subjects=()):
        self.kind = kind
        self.line = line
        self.end_line = end_line
        self.indent = indent
        self.list_role = list_role
        self.loop_depth = loop_depth
        self.try_depth = try_depth
        self.in_handler = in_handler
        self.before_kind = before_kind
        self.after_kind = after_kind
        self.n_stmts = n_stmts
        self.subjects = tuple(subjects)
        self.index = -1

    @property
    def loop_bucket(self):
        return "loop" if self.loop_depth else "flat"

    @property
    def site_id(self):
        return _digest("site", self.kind, self.line, self.end_line,
                       self.list_role, self.n_stmts)

    def sort_key(self):
        # Position first so the list reads down the function; `insert` before
        # `replace` at the same line so the cheaper shape is offered first.
        return (self.line, 0 if self.kind == "insert" else 1, self.end_line,
                self.list_role, self.n_stmts)

    def __repr__(self):                                   # pragma: no cover
        return "Site(%s, %d..%d, %s)" % (self.kind, self.line, self.end_line,
                                         self.list_role)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def symptom_subjects(symptom):
    """Identifier-shaped tokens in the symptom text, lowercased.

    A retrieval HINT for choosing which statement runs are worth replacing. It
    never admits or rejects anything; a run that matches no symptom token is
    still enumerable, it is just ordered after the ones that do.
    """
    return frozenset(t.lower() for t in _IDENT_RE.findall(symptom or ""))


def _stmt_names(nodes):
    out = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, ast.Attribute):
                out.add(n.attr)
            elif isinstance(n, ast.arg):
                out.add(n.arg)
    return out


def _bound_by(nodes):
    """Names the statements BIND (targets, loop vars, with/except aliases)."""
    out = set()
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.excepthandler) and n.name:
                out.add(n.name)
            elif isinstance(n, ast.alias):
                out.add((n.asname or n.name).split(".")[0])
    return out


def _stmt_lists(fn_node):
    """Every statement list in the function with its structural context.

    The docstring is skipped as a run member for the same reason
    `blocks.enumerate_insertion_points` skips it as a boundary: rewriting a
    docstring is not a behavioural fix and a family that offered it would spend
    samples proving that.
    """
    out = []

    def walk(body, list_role, loop_depth, try_depth, in_handler):
        skip = 0
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip = 1
        out.append((body[skip:], list_role, loop_depth, try_depth, in_handler,
                    body))
        for node in body:
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                walk(node.body, "loop_body", loop_depth + 1, try_depth,
                     in_handler)
                walk(node.orelse, "loop_else", loop_depth, try_depth,
                     in_handler)
            elif isinstance(node, ast.If):
                walk(node.body, "if_body", loop_depth, try_depth, in_handler)
                walk(node.orelse, "else_body", loop_depth, try_depth,
                     in_handler)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                walk(node.body, "with_body", loop_depth, try_depth, in_handler)
            elif isinstance(node, ast.Try):
                walk(node.body, "try_body", loop_depth, try_depth + 1,
                     in_handler)
                for h in node.handlers:
                    walk(h.body, "handler_body", loop_depth, try_depth, True)
                walk(node.orelse, "try_else", loop_depth, try_depth,
                     in_handler)
                walk(node.finalbody, "finally_body", loop_depth, try_depth,
                     in_handler)

    walk(fn_node.body, "fn_body", 0, 0, False)
    return out


def enumerate_replacement_spans(fn_node,
                                max_stmts=MAX_REPLACE_STATEMENTS):
    """Every contiguous statement run a block may REPLACE, in source order.

    This is the half a block TEMPLATE cannot express and the half most of the
    v3 board needs (14 of 21 golds are not pure additions). Bounded by
    ``max_stmts`` and by the enclosing statement list; never crosses a
    statement-list boundary, so the splice stays contiguous in the FILE too.
    """
    sites = []
    for body, list_role, loop_depth, try_depth, in_handler, full in \
            _stmt_lists(fn_node):
        for i in range(len(body)):
            for n in range(1, min(max_stmts, len(body) - i) + 1):
                run = body[i:i + n]
                lo = run[0].lineno - 1
                hi = getattr(run[-1], "end_lineno", run[-1].lineno) - 1
                if hi < lo:
                    continue
                idx_in_full = full.index(run[0])
                before = full[idx_in_full - 1] if idx_in_full > 0 else None
                after_i = idx_in_full + n
                after = full[after_i] if after_i < len(full) else None
                sites.append(Site(
                    "replace", lo, hi, B._node_indent(run[0]), list_role,
                    loop_depth, try_depth, in_handler,
                    B._stmt_kind(before), B._stmt_kind(after), n,
                    sorted(_stmt_names(run))))
    return sites


def enumerate_sites(fn_node, symptom="", limit=MAX_SITES,
                    max_stmts=MAX_REPLACE_STATEMENTS):
    """Every legal (position, shape) pair, DETERMINISTICALLY ordered.

    ``limit=None`` lifts the runtime budget. That is not a convenience: a
    coverage probe has no diagnostic line, so truncating at 8 would report
    "unreachable" for a function whose gold site is simply the ninth — the
    exact defect `blocks.TargetFacts` documents for its own point cap.
    """
    sites = []
    for p in B.enumerate_insertion_points(fn_node, limit=None):
        sites.append(Site("insert", p.line, p.line - 1, p.indent, p.list_role,
                          p.loop_depth, p.try_depth, p.in_handler,
                          p.before_kind, p.after_kind, 0, ()))
    sites.extend(enumerate_replacement_spans(fn_node, max_stmts))
    subjects = symptom_subjects(symptom)

    def rank(site):
        # Symptom overlap is a TIE-BREAKER inside a position, never a filter.
        # Filtering on it would let a badly-worded symptom delete the only
        # legal site, which is the silent-drop family (#181c) in miniature.
        overlap = len(subjects & {s.lower() for s in site.subjects})
        return (site.line, 0 if site.kind == "insert" else 1, -overlap,
                site.end_line, site.list_role, site.n_stmts)

    sites.sort(key=rank)
    for i, s in enumerate(sites):
        s.index = i
    return sites if limit is None else sites[:limit]


# ======================================================== 2. THE ENVELOPE
class Envelope:
    """The complete, checkable contract for ONE block hole.

    Everything in here is enforced by `validate_block`. Nothing in here is
    advisory: a field the validator does not read would be a rule the model is
    told about and then allowed to break, which is worse than no rule.
    """

    __slots__ = ("site_id", "kind", "role", "max_lines", "max_statements",
                 "max_nodes", "allowed_names", "allowed_calls", "must_bind",
                 "indent", "quote", "banned_calls", "removed_text")

    def __init__(self, site_id, kind, role, allowed_names, allowed_calls,
                 must_bind=(), indent=0, quote='"', max_lines=MAX_LINES,
                 max_statements=MAX_STATEMENTS, max_nodes=MAX_AST_NODES,
                 banned_calls=BANNED_CALLS, removed_text=""):
        self.site_id = site_id
        self.kind = kind
        self.role = role
        self.allowed_names = frozenset(allowed_names)
        self.allowed_calls = frozenset(allowed_calls)
        self.must_bind = tuple(sorted(must_bind))
        self.indent = indent
        self.quote = quote
        self.max_lines = max_lines
        self.max_statements = max_statements
        self.max_nodes = max_nodes
        self.banned_calls = frozenset(banned_calls)
        self.removed_text = removed_text

    def scope_names(self):
        """Allowed names MINUS the builtins: what this scope actually has."""
        return self.allowed_names - _BUILTIN_NAMES

    def rules_text(self):
        """The envelope as the model is told it. Deterministic, no model."""
        out = [
            "Rules for the block you write:",
            "- At most %d lines and at most %d statements."
            % (self.max_lines, self.max_statements),
            "- Plain Python statements only. No import, no def, no class, "
            "no lambda, no global or nonlocal, no exec or eval.",
            # SCOPE NAMES FIRST. Sorting the whole set alphabetically put 60
            # builtin exception classes in front and pushed every parameter and
            # local out of the list entirely — the model was told it could use
            # `ArithmeticError` and never shown `payload`. Builtins are named
            # as a class instead of enumerated; the validator still allows them
            # all, so this narrows the PROMPT, never the contract.
            "- Use only these names: %s (plus the Python builtins)"
            % (", ".join(sorted(self.scope_names())[:40]) or "(none)"),
            "- Any name you assign inside the block is yours to choose.",
            "- The payload is hole-relative: do not indent the first line; "
            "preserve indentation for nested statements.",
            "- It must parse as Python on its own.",
        ]
        if self.must_bind:
            out.append("- The block MUST still assign: %s"
                       % ", ".join(self.must_bind))
        return "\n".join(out)

    def __repr__(self):                                   # pragma: no cover
        return "Envelope(%s, %s, %d names)" % (self.kind, self.role,
                                               len(self.allowed_names))


def _quote_style(span_lines):
    text = "\n".join(span_lines)
    return "'" if text.count("'") > text.count('"') else '"'


def build_envelope(facts, site, symptom="", span_lines=(), fn_node=None,
                   role=None):
    """The rules for one site. CODE writes these; the model only obeys them.

    ``must_bind`` is what makes a REPLACEMENT safe: if the removed run bound a
    name a later statement reads, a replacement that drops the binding turns a
    working function into a NameError, and no amount of downstream witness
    running should have to discover that.
    """
    live = set(facts.params) | set(facts.module_names) | set(_BUILTIN_NAMES)
    live |= {n for n, ln in facts.assign_line.items() if ln <= site.line}
    must_bind = ()
    removed = ""
    max_lines, max_stmts, max_nodes = MAX_LINES, MAX_STATEMENTS, MAX_AST_NODES
    if site.kind == "replace" and fn_node is not None:
        run = _run_nodes(fn_node, site)
        bound = _bound_by(run)
        after = _names_read_after(fn_node, site.end_line)
        must_bind = sorted(bound & after)
        # A name the removed run bound is NOT live for the replacement.
        live -= (bound - set(facts.params) - set(facts.module_names)
                 - _BUILTIN_NAMES)
        if span_lines:
            removed = "\n".join(span_lines[site.line:site.end_line + 1])
        # A REPLACEMENT's budget is SPAN-RELATIVE: the model must reproduce
        # what it removes before it may add anything, so a flat 15-line cap
        # would reject a two-line change to a fourteen-line run. Measured on
        # the v3 board: a flat cap rejected TEN of nineteen expressible golds
        # whose gold text was mostly the statements they were replacing. The
        # cap is still a real bound — `MAX_REPLACE_STATEMENTS` bounds the span
        # itself, so this cannot grow into "rewrite the function".
        max_lines = (site.end_line - site.line + 1) + MAX_LINES
        max_stmts = len(run) + MAX_STATEMENTS
        max_nodes = sum(1 for n in run for _ in ast.walk(n)) + MAX_AST_NODES
    calls = set(facts.calls) | set(facts.module_names) | set(_BUILTIN_NAMES)
    calls |= {"%s.%s" % (p, a) for p in facts.params for a in facts.attributes}
    return Envelope(site.site_id, site.kind, role or classify_role(symptom),
                    live, calls, must_bind, site.indent,
                    _quote_style(span_lines) if span_lines else '"',
                    max_lines, max_stmts, max_nodes,
                    removed_text=removed)


def _run_nodes(fn_node, site):
    """The statement nodes a replacement site covers."""
    out = []
    for body, _role, _l, _t, _h, _full in _stmt_lists(fn_node):
        for node in body:
            lo = node.lineno - 1
            hi = getattr(node, "end_lineno", node.lineno) - 1
            if site.line <= lo and hi <= site.end_line:
                out.append(node)
    # Keep only the outermost nodes: a nested statement is already covered.
    top = []
    for node in sorted(out, key=lambda n: (n.lineno, -n.end_lineno)):
        if top and node.lineno - 1 <= getattr(top[-1], "end_lineno",
                                              top[-1].lineno) - 1:
            continue
        top.append(node)
    return top


def _names_read_after(fn_node, end_line):
    out = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                and n.lineno - 1 > end_line:
            out.add(n.id)
    return out


# ====================================================== 3. THE AUTHOR CALL
def build_author_prompt(site, envelope, span_lines, symptom):
    """The complete model input for ONE block hole. Deterministic.

    Everything the model may know: the symptom, the enclosing function with the
    hole marked, the envelope rules in plain words, and the role verb. It does
    not receive the gold, the tests, the diff, or an instruction to "recover"
    anything — the same discipline as `holes.build_fill_prompt`.
    """
    marked = _marked_source(site, span_lines)
    lines = ["Problem: %s" % (symptom or "unspecified").strip(), ""]
    if site.kind == "replace":
        lines.append("Rewrite the lines marked <<<BLOCK>>> in this function.")
    else:
        lines.append("Write the statements that belong at <<<BLOCK>>> in this "
                     "function.")
    lines += ["", marked, ""]
    verb = ROLE_VERBS.get(envelope.role, ROLE_VERBS["other"])
    lines.append("What the block must do: %s" % verb)
    if site.kind == "replace" and envelope.removed_text.strip():
        lines += ["", "The lines being replaced are:", envelope.removed_text]
    lines += ["", envelope.rules_text(), "",
              "Answer with the block's Python lines alone. No explanation, "
              "no code fences."]
    return "\n".join(lines)


def _marked_source(site, span_lines):
    out = list(span_lines)
    pad = " " * site.indent
    if site.kind == "replace":
        lo, hi = site.line, site.end_line
        if not (0 <= lo <= hi < len(out)):
            return "\n".join(span_lines)
        return "\n".join(out[:lo] + [pad + "<<<BLOCK>>>"] + out[hi + 1:])
    li = site.line
    if not (0 <= li <= len(out)):
        return "\n".join(span_lines)
    return "\n".join(out[:li] + [pad + "<<<BLOCK>>>"] + out[li:])


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*$")


def decode_block(raw):
    """Model text -> block source, or ``None``.

    Strips a think block and one code fence. It does NOT try to find code
    inside prose: a model that answered with a paragraph failed the protocol,
    and salvaging it would make the pass rate a property of this function.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip("\n").rstrip()
    if text.lstrip().startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            return None
        # `.strip()` here would eat the block's INDENTATION along with the
        # newline, and the indent gate would then reject a block the model got
        # right. Only whole leading blank lines may go.
        text = text[end + len("</think>"):].lstrip("\r\n")
    lines = text.split("\n")
    if lines and _FENCE_RE.match(lines[0].strip()):
        lines = lines[1:]
        for i, ln in enumerate(lines):
            if _FENCE_RE.match(ln.strip()):
                lines = lines[:i]
                break
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if not lines:
        return None
    return "\n".join(lines)


def _first_code_indent(text):
    """Return the first non-comment code line's leading-space count."""
    for line in text.split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            return len(line) - len(line.lstrip())
    return None


def normalize_block(text, indent):
    """Canonicalize a decoded fill to the site's source indentation.

    The author contract is hole-relative, but a model may still copy the
    absolute indentation shown in the surrounding source. Removing only the
    common outer indentation and then adding the code-owned site prefix makes
    both representations equivalent without changing their nested structure.
    Validation remains strict and is intentionally performed afterwards.
    """
    if not isinstance(text, str):
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = textwrap.dedent(normalized)
    prefix = " " * max(0, int(indent))
    return "\n".join(
        (prefix + line) if line.strip() else ""
        for line in normalized.split("\n")
    )


# ========================================================= 4. VALIDATION
class BlockVerdict:
    """``ok`` plus the ONE rule that rejected it, plus soft style flags."""

    __slots__ = ("ok", "reason", "detail", "style_flags", "stmts", "lines",
                 "nodes")

    def __init__(self, ok, reason="", detail="", style_flags=(), stmts=0,
                 lines=0, nodes=0):
        self.ok = ok
        self.reason = reason
        self.detail = detail
        self.style_flags = tuple(style_flags)
        self.stmts = stmts
        self.lines = lines
        self.nodes = nodes

    def __repr__(self):                                   # pragma: no cover
        return "BlockVerdict(%s, %s)" % (self.ok, self.reason or "-")


def _reject(reason, detail="", flags=()):
    return BlockVerdict(False, reason, detail, flags)


def block_bound_names(nodes):
    """Names the block itself binds — these need no envelope permission."""
    out = _bound_by(nodes)
    for node in nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        out.add(t.id)
    return out


def style_report(text, envelope):
    """SOFT style observations. Never a rejection — see `STYLE_QUOTE`."""
    flags = []
    body = textwrap.dedent(text)
    other = "'" if envelope.quote == '"' else '"'
    if other in body and envelope.quote not in body:
        flags.append(STYLE_QUOTE)
    if any(len(ln) > MAX_STYLE_LINE for ln in text.split("\n")):
        flags.append(STYLE_LINE_LENGTH)
    return tuple(flags)


def validate_block(text, envelope):
    """THE AUTHORITY. Fail-closed: an unrecognised shape is a rejection.

    Order matters and is cheap-to-expensive: protocol, size, parse, then the
    semantic gates. Every branch names a distinct reason so the telemetry can
    say WHICH rule the model broke rather than "it was rejected".
    """
    if not isinstance(text, str) or not text.strip():
        return _reject(REJECT_EMPTY)
    if "<<<BLOCK>>>" in text or "__hole" in text or "__BH" in text:
        return _reject(REJECT_PLACEHOLDER)

    raw_lines = text.rstrip("\n").split("\n")
    code_lines = [ln for ln in raw_lines
                  if ln.strip() and not ln.strip().startswith("#")]
    if not code_lines:
        return _reject(REJECT_EMPTY)
    if len(raw_lines) > envelope.max_lines:
        return _reject(REJECT_LINES, "%d lines" % len(raw_lines))

    # Indentation is a HARD gate: a block whose first line does not sit at the
    # site's column is not the block the harness asked for, and splicing it
    # would either shift the following statement into the wrong suite or fail
    # to parse two hundred lines away from the cause.
    first = code_lines[0]
    if len(first) - len(first.lstrip()) != envelope.indent:
        return _reject(REJECT_INDENT,
                       "first line at col %d, site is %d"
                       % (len(first) - len(first.lstrip()), envelope.indent))

    body = textwrap.dedent(text)
    nodes = B.parse_block(body)
    if nodes is None:
        return _reject(REJECT_PARSE)
    if len(nodes) > envelope.max_statements:
        return _reject(REJECT_STATEMENTS, "%d statements" % len(nodes))
    n_nodes = sum(1 for n in nodes for _ in ast.walk(n))
    if n_nodes > envelope.max_nodes:
        return _reject(REJECT_NODES, "%d nodes" % n_nodes)

    for types, reason in BANNED_NODE_REASONS:
        for n in nodes:
            for c in ast.walk(n):
                if isinstance(c, types):
                    return _reject(reason, type(c).__name__)

    bound = block_bound_names(nodes)
    for n in nodes:
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                head = B.dotted(c.func)
                root = head.split(".")[0] if head else None
                if root in envelope.banned_calls:
                    return _reject(REJECT_UNKNOWN_CALL, root)
                if head and root not in bound \
                        and head not in envelope.allowed_calls \
                        and root not in envelope.allowed_names:
                    return _reject(REJECT_UNKNOWN_CALL, head)
            elif isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                if c.id not in bound and c.id not in envelope.allowed_names:
                    return _reject(REJECT_UNKNOWN_NAME, c.id)

    missing = [n for n in envelope.must_bind if n not in bound]
    if missing:
        return _reject(REJECT_MUST_BIND, ",".join(missing))

    return BlockVerdict(True, "", "", style_report(text, envelope),
                        len(nodes), len(raw_lines), n_nodes)


# ==================================================== 5. SAMPLING (model)
def _seed(site_id, i):
    return int(_digest(site_id, i)[:8], 16) % (2 ** 31)


def sample_blocks(site, envelope, span_lines, symptom, complete=None,
                  samples=BLOCK_SAMPLES, budget=None):
    """``(blocks, telemetry)`` — up to ``samples`` DEDUPED, VALID blocks.

    ``complete(prompt, answer_tokens=..., seed=...) -> str`` is the injected
    model seam, so every test here runs offline against a stub and the protocol
    is provable without a GPU.

    A sample that fails validation is COUNTED and DISCARDED. It is never
    retried into existence: retrying until something valid appears would make
    pass@N dishonest and the spend unbounded (`holes.sample_fills`).
    """
    tel = {"requested": 0, "returned": 0, "valid": 0, "rejected": {},
           "budget_exhausted": False, "call_failed": 0, "style_flags": {},
           "deduped": 0}
    if complete is None:
        return (), tel
    prompt = build_author_prompt(site, envelope, span_lines, symptom)
    out, seen = [], set()
    for i in range(max(1, samples)):
        if budget is not None and not budget.take():
            tel["budget_exhausted"] = True
            break
        tel["requested"] += 1
        try:
            raw = complete(prompt, answer_tokens=ANSWER_TOKENS,
                           seed=_seed(site.site_id, i))
        except Exception as exc:          # law 1: a dead provider is REPORTED
            tel["call_failed"] += 1
            tel.setdefault("errors", []).append(type(exc).__name__)
            continue
        text = decode_block(raw)
        if text is None:
            tel["rejected"][REJECT_PROTOCOL] = \
                tel["rejected"].get(REJECT_PROTOCOL, 0) + 1
            continue
        tel["returned"] += 1
        raw_indent = _first_code_indent(text)
        canonical = normalize_block(text, envelope.indent)
        tel.setdefault("normalization", []).append({
            "raw_indent": raw_indent,
            "canonical_indent": _first_code_indent(canonical),
            "normalization_applied": canonical != text,
        })
        verdict = validate_block(canonical, envelope)
        if not verdict.ok:
            tel["rejected"][verdict.reason] = \
                tel["rejected"].get(verdict.reason, 0) + 1
            continue
        key = ast_fingerprint(canonical)
        if key in seen:
            tel["deduped"] += 1
            continue
        seen.add(key)
        for f in verdict.style_flags:
            tel["style_flags"][f] = tel["style_flags"].get(f, 0) + 1
        out.append(canonical)
    tel["valid"] = len(out)
    return tuple(out), tel


def ast_fingerprint(text):
    """Canonical AST digest — the dedup key. Whitespace is not a difference."""
    nodes = B.parse_block(textwrap.dedent(text))
    if nodes is None:
        return None
    return _digest("ast", B.canonical_dump(nodes))


# ================================================ 6. CANDIDATE CONSTRUCTION
def make_block_edit(span_lines, site, block_text, provenance="block_hole"):
    """One atomic block edit as an `edit_sets.EditSet`, or ``None``.

    INSERT is expressed as a replacement of the anchor line by
    ``block + anchor`` — the same trick `blocks.make_insertion` uses, and the
    reason there is no state in which half the block landed.

    REPLACE is expressed as a replacement of the covered lines. Same splice,
    same atomicity, and it is the shape a block TEMPLATE cannot produce.
    """
    if not isinstance(block_text, str) or not block_text.strip():
        return None
    if site.kind == "insert":
        li = site.line
        if not (0 <= li < len(span_lines)):
            return None
        anchor = span_lines[li]
        new_block = block_text + "\n" + anchor
        edit = es.SpanEdit(anchor, new_block, role="block_hole_insert")
        try:
            return es.EditSet((li, li), anchor, new_block, [edit],
                              "%s:insert:%s" % (provenance, site.site_id),
                              sorted(_stmt_names(
                                  B.parse_block(textwrap.dedent(block_text))
                                  or [])))
        except Exception:
            return None
    lo, hi = site.line, site.end_line
    if not (0 <= lo <= hi < len(span_lines)):
        return None
    old = "\n".join(span_lines[lo:hi + 1])
    edit = es.SpanEdit(old, block_text, role="block_hole_replace")
    try:
        return es.EditSet((lo, hi), old, block_text, [edit],
                          "%s:replace:%s" % (provenance, site.site_id),
                          sorted(_stmt_names(
                              B.parse_block(textwrap.dedent(block_text))
                              or [])))
    except Exception:
        return None


# ================================================= 7. NEAR-MISS EXTENSION
NEAR_MISS_NONE = ""
#: The witness agrees, the AST does not. A DIFFERENT correct block — the class
#: the strict normalized-AST scorer must not silently record as "wrong", or the
#: board reads as a generator failure when the generator succeeded.
NEAR_MISS_BLOCK_SEMANTIC = "near_miss_block_semantic"
#: The AST agrees but the block landed at a different legal position.
NEAR_MISS_BLOCK_POSITION = "near_miss_block_position"


def classify_block_near_miss(got_norm, gold_norm, behavior_match=None,
                             got_block_ast=None, gold_block_ast=None):
    """Near-miss class for one applied block candidate.

    ``behavior_match`` is the WITNESS's verdict and is the only thing that may
    promote a mismatch to ``near_miss_block_semantic``. Passing ``True`` from a
    heuristic instead of an executed witness would be inventing evidence, which
    is exactly the failure this classification exists to expose.
    """
    if got_norm is not None and got_norm == gold_norm:
        return NEAR_MISS_NONE                       # exact: not a near miss
    if behavior_match is True:
        return NEAR_MISS_BLOCK_SEMANTIC
    if (got_block_ast is not None and gold_block_ast is not None
            and got_block_ast == gold_block_ast):
        return NEAR_MISS_BLOCK_POSITION
    return NEAR_MISS_NONE


# =========================================================== 8. THE SEAM
#: THE ONE FUNCTION THE FIXER MAY CALL.
#:
#: Wiring is the NEXT lane's work and is deliberately NOT done here, because
#: this lane may not measure live fill quality (`BLOCK_HOLE_PUNTS`). When the
#: live referendum flips it, it is these four lines and nothing else:
#:
#:   1. fixer.py FAMILY_MAP (the only grant path):
#:          "block_hole": "block_hole",
#:   2. gym_config.py fix.families: append "block_hole" — AND flip
#:      `block_holes.BLOCK_HOLE_ENABLED`, which fails closed until then.
#:   3. the enumerator arm, next to the existing family calls:
#:          import block_holes
#:          cands = block_holes.block_hole_candidates(
#:              span, start, file_lines, symptom=symptom, complete=complete)
#:   4. menu pressure (w42): screen <=8 candidates, expose <=2 block entries,
#:      phrase each from the ROLE grammar, never from raw source.
#:
#: Every returned candidate is already an `edit_sets.EditSet`, so the existing
#: `edit_sets.apply_candidate(lines, mut)` site applies it unchanged and `None`
#: still means skip/fail-closed.
BLOCK_HOLE_SEAM = (
    "block_holes.block_hole_candidates(span_lines, span_start, file_lines, "
    "symptom='', complete=None, fn_name=None, samples=BLOCK_SAMPLES, "
    "site_limit=MAX_SITES, budget=None, enabled=None, telemetry=None) "
    "-> list[edit_sets.EditSet]")


def enclosing_scope_names(file_lines, fn_name):
    """Names bound by the ENCLOSING functions of ``fn_name``, if any.

    `blocks.TargetFacts` sees the target function and the module, and nothing
    between them. For a NESTED function that is wrong: every parameter and
    local of the enclosing scope is legally readable at the site, and an
    envelope that omits them rejects a perfectly legal block with
    `REJECT_UNKNOWN_NAME`. Measured on the v3 board: `invoke_fallback` is a
    closure and its gold calls a helper bound by its parent.

    Returns an empty set when the file does not parse or the function is not
    nested — never raises, never widens beyond the real ancestor chain.
    """
    if not file_lines:
        return frozenset()
    try:
        tree = ast.parse("\n".join(file_lines))
    except SyntaxError:
        return frozenset()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == fn_name:
            target = node
            break
    if target is None:
        return frozenset()
    out = set()
    cur = parents.get(target)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = cur.args
            out.update(p.arg for p in
                       (list(getattr(a, "posonlyargs", [])) + list(a.args)
                        + ([a.vararg] if a.vararg else [])
                        + list(a.kwonlyargs)
                        + ([a.kwarg] if a.kwarg else [])))
            for n in ast.walk(cur):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)) and n is not cur:
                    out.add(n.name)
        cur = parents.get(cur)
    return frozenset(out)


def target_facts(span_lines, file_lines=None, fn_name=None, point_limit=None):
    """`blocks.TargetFacts` for a span, or ``None``. Shared by seam + probe."""
    src = textwrap.dedent("\n".join(span_lines))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None
    fn_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and (fn_name is None or node.name == fn_name):
            fn_node = node
            break
    if fn_node is None:
        return None, None
    mod = B.module_level_names("\n".join(file_lines)) if file_lines \
        else frozenset()
    # Enclosing-scope names ride in through `module_names` because that is the
    # channel `TargetFacts` already treats as "resolvable but not local". They
    # are genuinely in scope; nothing here widens the set beyond the real
    # ancestor chain.
    mod = mod | enclosing_scope_names(file_lines, fn_node.name)
    return fn_node, B.TargetFacts(fn_node, mod, point_limit=point_limit)


def block_hole_candidates(span_lines, span_start=0, file_lines=None,
                          symptom="", complete=None, fn_name=None,
                          samples=BLOCK_SAMPLES, site_limit=MAX_SITES,
                          budget=None, enabled=None, telemetry=None):
    """THE SEAM. Bounded, deterministic block-hole candidates for a span.

    Returns ``[]`` — never raises — when the family is disabled, when the span
    does not parse, or when no model seam was supplied. Fail-closed everywhere:
    a family that cannot run must produce nothing, not something plausible.

    ``span_start`` is accepted and unused, for signature parity with the other
    families' call sites (`blocks.block_insertion_candidates`); module-level
    names come from ``file_lines``. ``telemetry``, when a list is passed,
    collects ``(site_id, sample telemetry)`` per site — the wiring lane needs
    the rejection tallies to phrase menu entries and to see a dead provider.
    """
    if not (BLOCK_HOLE_ENABLED if enabled is None else enabled):
        return []
    if complete is None:
        return []
    fn_node, facts = target_facts(span_lines, file_lines, fn_name)
    if facts is None:
        return []
    shift = B._dedent_shift(span_lines)
    out = []
    for site in enumerate_sites(fn_node, symptom, limit=site_limit):
        site.indent += shift
        env = build_envelope(facts, site, symptom, span_lines, fn_node)
        texts, tel = sample_blocks(site, env, span_lines, symptom, complete,
                                   samples=samples, budget=budget)
        if telemetry is not None:
            telemetry.append((site.site_id, tel))
        for text in texts:
            cand = make_block_edit(span_lines, site, text)
            if cand is not None:
                out.append(cand)
    return out
