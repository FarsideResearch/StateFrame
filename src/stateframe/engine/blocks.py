"""BLOCK TEMPLATES — the offline half of the hybrid block-synthesis design.

WHAT THIS IS
------------
The v3 structural board's remaining wall is not "the model cannot write a
token", it is "the enumerator cannot build a STATEMENT BLOCK". `holes.py`
substitutes leaves into a structure that already exists; nothing in the tree
ADDS one. This module is lane A of `/tmp/solq/w39/a_block_synthesis.md`: a
corpus of 1-15 line statement blocks mined from real fix history
(`probes/block_miner.py`), each stored as a typed-hole skeleton that CODE — not
a model — unifies against a target function's symbol table to produce an atomic
block-insertion candidate.

WHAT IT IS NOT
--------------
It is not a fix path yet. Wiring belongs to the next lane and reaches this
module through exactly ONE function, `block_insertion_candidates` (see
`BLOCK_FIXER_SEAM`). Nothing here imports `fixer`.

DELIBERATE DEVIATIONS FROM THE DESIGN (measured, not silent)
------------------------------------------------------------
The two design notes were written without repo access. Three of their
assumptions do not hold here, and pretending otherwise would produce a library
whose numbers mean nothing:

1. "approximately 40k general commits". MEASURED: `lab-repo` carries 245
   commits (9 touching `backend/`) and `mission-control` carries 1720. The
   mining envelope is therefore ~1.9k commits, not 40k, and the yield bands in
   the design (300..900 promotable clusters) are off by more than an order of
   magnitude. `probes/block_miner.py` reports its own coverage rather than
   quoting the design's.

2. "resolved: base witness fails, post witness passes". There is no per-commit
   witness executor for historical AHA/MC revisions, and building one would
   mean running each repo's suite at ~2k revisions. Every mined episode is
   therefore the design's `candidate` class. `witness_family` on a template is
   a STATIC obligation partition derived from the block's role (the design's
   §4 partitions), never an executed result, and `promoted` is a purely
   structural predicate. Nothing in this module may be read as "witnessed".

3. `Exit<...>` and `Predicate<...>` as free holes. Both are kept as CHECKED
   CONSTANTS instead: the exit kind and the comparison operator/polarity ride
   in the effect/precondition signature and are verified legal at the insertion
   point, while only the operands are holes. A free exit hole would let
   unification turn `continue` into `return`, which is a different fix, and the
   design's own clustering gates forbid merging those two shapes anyway.

Everything is deterministic: template ids are content hashes, retrieval is a
sorted intersection, and binding enumerates a bounded product in a fixed order.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import re
import textwrap

import edit_sets as es

_BUILTIN_NAMES = frozenset(dir(builtins))

SCHEMA_VERSION = 3

# --------------------------------------------------------------- policy caps
#: A block wider than this is a composite diagnostic, not one template
#: (`a_block_mining.md` §1 "Initial limits", widened from 6 statements to the
#: 15 LINES the synthesis note names, because the v3 golds carry comment lines
#: inside the added block and a statement cap alone would admit a 40-line one).
MAX_BLOCK_LINES = 15
MAX_BLOCK_STATEMENTS = 6
MAX_BLOCK_AST_NODES = 96

#: Holes one template may carry. Deliberately wider than
#: `holes.MAX_HOLES_PER_TEMPLATE` (3): that ceiling bounds a SUBSTITUTION into
#: an existing structure, where every extra hole multiplies against a menu the
#: model must choose from. Here the structure itself is donated and the holes
#: are bound by a symbol table, so the ceiling that matters is the binding
#: product, which `MAX_BINDING_CANDIDATES` bounds directly.
MAX_BLOCK_HOLES = 8

#: Templates evaluated per insertion point (`a_block_mining.md` §3).
MAX_TEMPLATES_PER_POINT = 16
#: Insertion points inspected per function (design: "up to 60 legal points").
MAX_INSERTION_POINTS = 60
#: Distinct bindings one template may contribute at one point.
MAX_BINDING_CANDIDATES = 4
#: Total candidates the seam returns.
BLOCK_CANDIDATE_LIMIT = 48

ROLES = ("loop_guard", "revalidate", "sanitize", "guard", "other")
#: `a_block_mining.md` §1 "Primary-role ordering".
ROLE_PRIORITY = {r: i for i, r in enumerate(ROLES)}

HOLE_KINDS = (
    "Name<Input>",
    "Name<BoundLocal>",
    "Name<Receiver>",
    #: A name resolvable in neither the donor's parameters, its locals, nor its
    #: module scope: a CLOSURE free variable. The design's list has no entry for
    #: it and rejecting the block outright lost real donors (`invoke_fallback`
    #: on the v3 board is a nested `def` reading three enclosing names), so it
    #: is typed rather than dropped. Its binding domain is params + live locals,
    #: i.e. strictly narrower than the donor's scope.
    "Name<Free>",
    "Call<RegisteredAPI>",
    "Literal<const>",
)

EXIT_KINDS = ("return", "raise", "continue", "break", "none")

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "block_templates.json")

#: Declared, never silently missing. Read alongside any reachability number.
BLOCK_PUNTS = {
    "no_witness_execution":
        "No historical episode is witness-verified (deviation 2 above). Every "
        "template is the design's `candidate` class; promotion is structural.",
    "exit_is_a_constant":
        "Exit kind and comparison operator are checked constants, not holes "
        "(deviation 3). A gold whose only difference from a template is "
        "`continue` vs `return` is a MISS here, by choice.",
    "single_block_only":
        "A gold needing two non-adjacent inserted blocks is not expressible: "
        "one template is one contiguous sibling run. The design's composite "
        "fix record is recorded by the miner and never promoted.",
    "no_typed_binding_domain":
        "Binding checks name availability and scope, not value class. A "
        "template wanting a dict-shaped local will bind to any live local.",
}


# ===================================================================== 1. AST
def parse_block(text):
    """Parse a bare statement block, or ``None``.

    A mined block is a fragment of a function body: it may open with `continue`
    or `return`, neither of which parses as a module. The wrappers are tried in
    a fixed order and the wrapper's own body is returned, so a caller never
    sees the scaffolding.
    """
    body = textwrap.dedent(text)
    if not body.strip():
        return None
    try:
        return ast.parse(body).body or None
    except SyntaxError:
        pass
    ind = textwrap.indent(body, "    ")
    for wrap in ("while True:\n", "async def _w():\n", "def _w():\n"):
        try:
            mod = ast.parse(wrap + ind)
        except SyntaxError:
            continue
        return mod.body[0].body or None
    return None


def canonical_dump(nodes):
    """Location-free AST dump of a statement list."""
    mod = ast.Module(body=list(nodes), type_ignores=[])
    return ast.dump(mod, annotate_fields=True, include_attributes=False)


def dotted(node):
    """`a.b.c` for a Name/Attribute chain, else ``None``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ============================================================ 2. TYPED HOLES
class Hole:
    """One typed hole in a block skeleton."""

    __slots__ = ("hid", "kind", "subtype", "default", "placeholder")

    def __init__(self, hid, kind, subtype="", default=None):
        if kind not in HOLE_KINDS:
            raise ValueError("unknown hole kind %r" % (kind,))
        self.hid = hid
        self.kind = kind
        self.subtype = subtype
        self.default = default
        self.placeholder = "__BH%d__" % hid

    def to_json(self):
        return {"hid": self.hid, "kind": self.kind, "subtype": self.subtype,
                "default": self.default}

    @staticmethod
    def from_json(d):
        return Hole(d["hid"], d["kind"], d.get("subtype", ""),
                    d.get("default"))

    def __eq__(self, other):
        return (isinstance(other, Hole) and self.hid == other.hid
                and self.kind == other.kind and self.subtype == other.subtype
                and self.default == other.default)

    def __hash__(self):
        return hash((self.hid, self.kind, self.subtype, repr(self.default)))

    def __repr__(self):                                   # pragma: no cover
        return "Hole(%d, %s, %r)" % (self.hid, self.kind, self.default)


# ========================================================= 3. THE TEMPLATE
class BlockTemplate:
    """A mined statement block, abstracted to typed holes.

    ``skeleton`` is dedented source with `__BHn__` placeholders in the hole
    positions; it always parses through `parse_block`, which is checked at
    construction so a template that cannot become a candidate is never stored.
    """

    __slots__ = ("tid", "role", "role_mask", "skeleton", "holes",
                 "precondition", "effect", "witness_family", "provenance",
                 "n_instances", "lineages", "modules")

    def __init__(self, role, role_mask, skeleton, holes, precondition, effect,
                 witness_family=None, provenance=(), n_instances=1,
                 lineages=(), modules=()):
        if role not in ROLES:
            raise ValueError("unknown role %r" % (role,))
        if parse_block(skeleton) is None:
            raise ValueError("skeleton does not parse")
        self.role = role
        self.role_mask = tuple(sorted(set(role_mask) | {role}))
        self.skeleton = skeleton
        self.holes = tuple(holes)
        if len(self.holes) > MAX_BLOCK_HOLES:
            raise ValueError("template over the %d-hole ceiling"
                             % MAX_BLOCK_HOLES)
        self.precondition = dict(precondition)
        self.effect = dict(effect)
        self.witness_family = dict(witness_family or
                                   witness_obligations(role))
        self.provenance = tuple(provenance)
        self.n_instances = n_instances
        self.lineages = tuple(sorted(set(lineages)))
        self.modules = tuple(sorted(set(modules)))
        self.tid = self.exact_key()

    # ---------------------------------------------------------------- keys
    def exact_key(self):
        """`a_block_mining.md` §2 "Exact template key" — content identity.

        Literal VALUES are inside the skeleton, so `x is None`, `if not x` and
        `x < 3` cannot collapse into one id, which is the property the design
        calls out by name.
        """
        return _digest(SCHEMA_VERSION, self.role, self.role_mask,
                       self.skeleton, [h.to_json() for h in self.holes],
                       sorted(self.precondition.items(), key=repr),
                       sorted(self.effect.items(), key=repr))

    @property
    def n_lines(self):
        return len(self.skeleton.rstrip("\n").split("\n"))

    @property
    def promoted(self):
        """STRUCTURAL promotion only — see deviation 2. Never "witnessed"."""
        return (self.n_instances >= 3 and len(self.lineages) >= 2
                and len(self.modules) >= 2)

    # ---------------------------------------------------------------- json
    def to_json(self):
        return {
            "tid": self.tid,
            "schema": SCHEMA_VERSION,
            "role": self.role,
            "role_mask": list(self.role_mask),
            "skeleton": self.skeleton,
            "holes": [h.to_json() for h in self.holes],
            "precondition_signature": self.precondition,
            "effect_signature": self.effect,
            "witness_family": self.witness_family,
            "provenance": [dict(p) for p in self.provenance],
            "n_instances": self.n_instances,
            "lineages": list(self.lineages),
            "modules": list(self.modules),
            "promoted": self.promoted,
            "evidence_class": "candidate",      # never "resolved" — see punts
        }

    @staticmethod
    def from_json(d):
        return BlockTemplate(
            d["role"], d.get("role_mask", [d["role"]]), d["skeleton"],
            [Hole.from_json(h) for h in d.get("holes", [])],
            d.get("precondition_signature", {}), d.get("effect_signature", {}),
            d.get("witness_family"),
            tuple(dict(p) for p in d.get("provenance", [])),
            d.get("n_instances", 1), d.get("lineages", ()),
            d.get("modules", ()))

    def __repr__(self):                                   # pragma: no cover
        return "BlockTemplate(%s, %s, %dL, %dH)" % (self.tid, self.role,
                                                    self.n_lines,
                                                    len(self.holes))


def witness_obligations(role):
    """The design's §4 finite partitions. STATIC — nothing here was executed."""
    partitions = {
        "guard": ["none", "falsey_non_none", "truthy", "malformed",
                  "threshold_neighbor"],
        "sanitize": ["dirty", "canonical", "idempotent_repeat", "malformed"],
        "revalidate": ["valid_no_change", "valid_then_mutated_invalid",
                       "initially_invalid", "refresh_failure"],
        "loop_guard": ["empty", "trigger_first", "trigger_middle",
                       "trigger_last", "ordinary_item", "count_boundary"],
        "other": ["nominal", "boundary"],
    }
    mutations = {
        "guard": ["remove_block", "invert_predicate", "null_test_to_truthy",
                  "move_after_sink"],
        "sanitize": ["remove_block", "drop_normalization", "move_after_sink"],
        "revalidate": ["remove_block", "change_validator_receiver",
                       "move_after_sink"],
        "loop_guard": ["remove_block", "invert_predicate",
                       "continue_to_fallthrough", "break_to_continue"],
        "other": ["remove_block"],
    }
    return {"partitions": partitions[role], "mutations": mutations[role],
            "executed": False}


# ================================================== 4. TARGET-SIDE ANALYSIS
class InsertionPoint:
    """One legal statement-list boundary inside the target function."""

    __slots__ = ("index", "line", "indent", "list_role", "loop_depth",
                 "try_depth", "in_handler", "before_kind", "after_kind")

    def __init__(self, index, line, indent, list_role, loop_depth, try_depth,
                 in_handler, before_kind, after_kind):
        self.index = index
        self.line = line                 # span-relative line to insert BEFORE
        self.indent = indent
        self.list_role = list_role
        self.loop_depth = loop_depth
        self.try_depth = try_depth
        self.in_handler = in_handler
        self.before_kind = before_kind
        self.after_kind = after_kind

    @property
    def loop_bucket(self):
        return "loop" if self.loop_depth else "flat"

    @property
    def try_bucket(self):
        return "handler" if self.in_handler else ("try" if self.try_depth
                                                  else "flat")

    def key(self):
        return (self.list_role, self.loop_bucket, self.try_bucket,
                self.before_kind, self.after_kind)

    def __repr__(self):                                   # pragma: no cover
        return "InsertionPoint(line=%d, %s, %s)" % (self.line, self.list_role,
                                                    self.loop_bucket)


def _stmt_kind(node):
    return type(node).__name__ if node is not None else "-"


def function_node(src, fn_name):
    """The named function's AST node inside ``src``, or ``None``."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == fn_name:
            return node
    return None


def enumerate_insertion_points(fn_node, limit=MAX_INSERTION_POINTS):
    """Every legal statement-list boundary in the function, in source order.

    Deterministic and bounded. The docstring statement is never a boundary
    predecessor position (the design forbids crossing a docstring boundary),
    so a block can never be inserted above a function docstring.
    """
    points = []

    def walk(body, list_role, loop_depth, try_depth, in_handler):
        skip = 0
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip = 1
        for i in range(skip, len(body) + 1):
            before = body[i - 1] if i > 0 else None
            after = body[i] if i < len(body) else None
            anchor = after if after is not None else before
            if anchor is None:
                continue
            line = (anchor.lineno - 1) if after is not None \
                else getattr(anchor, "end_lineno", anchor.lineno)
            points.append(InsertionPoint(
                len(points), line, _node_indent(anchor), list_role, loop_depth,
                try_depth, in_handler, _stmt_kind(before), _stmt_kind(after)))
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
    points.sort(key=lambda p: (p.line, p.list_role))
    for i, p in enumerate(points):
        p.index = i
    return points if limit is None else points[:limit]


def _node_indent(node):
    return getattr(node, "col_offset", 0)


class TargetFacts:
    """Everything unification is allowed to know about one target function.

    Computed ONCE per function (`a_block_mining.md` §3) and shared across every
    insertion point, which is what keeps retrieval off the O(all-templates)
    path.
    """

    __slots__ = ("fn_name", "params", "is_async", "assigned", "assign_line",
                 "module_names", "calls", "attributes", "points", "fn_node")

    def __init__(self, fn_node, module_names=(),
                 point_limit=MAX_INSERTION_POINTS):
        self.fn_node = fn_node
        self.fn_name = fn_node.name
        self.is_async = isinstance(fn_node, ast.AsyncFunctionDef)
        a = fn_node.args
        self.params = tuple(p.arg for p in
                            (list(getattr(a, "posonlyargs", [])) + list(a.args)
                             + ([a.vararg] if a.vararg else [])
                             + list(a.kwonlyargs)
                             + ([a.kwarg] if a.kwarg else [])))
        self.assigned = {}
        self.attributes = set()
        self.calls = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.assigned.setdefault(node.id, node.lineno)
            elif isinstance(node, ast.Attribute):
                self.attributes.add(node.attr)
            elif isinstance(node, ast.Call):
                d = dotted(node.func)
                if d:
                    self.calls.add(d)
        self.assign_line = dict(self.assigned)
        self.module_names = frozenset(module_names)
        # `point_limit=None` lifts the runtime locality budget. That budget
        # exists because a FIXER has a diagnostic line and only needs the
        # boundaries near it; an offline COVERAGE probe has no diagnostic and
        # truncating at 60 would report "unreachable" for a function whose gold
        # position is simply line 61. Measured: row 3 of the v3 board missed
        # for exactly that reason before this parameter existed.
        self.points = enumerate_insertion_points(fn_node, point_limit)

    def locals_live_at(self, point):
        """Locals bound strictly BEFORE the insertion line."""
        return tuple(sorted(n for n, ln in self.assign_line.items()
                            if ln <= point.line))

    def name_available(self, name, point):
        # BUILTINS ARE ALWAYS AVAILABLE. Omitting them made every template that
        # raises `ValueError` unretrievable in a target whose module never
        # names it — which is most of them — and the symptom was an empty
        # candidate list with no error anywhere.
        return (name in self.params or name in self.module_names
                or name in _BUILTIN_NAMES
                or self.assign_line.get(name, 10 ** 9) <= point.line)

    def call_available(self, call):
        """Is a dotted API name resolvable from this function?"""
        if call in self.calls:
            return True
        root = call.split(".")[0]
        return (root in self.module_names or root in self.params
                or root in self.assign_line or root in _BUILTIN_NAMES)


def module_level_names(src):
    """Imports, module-level defs/classes and module-level assignments."""
    out = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return frozenset()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out.add(node.name)
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in ([node.target] if isinstance(node, ast.AnnAssign)
                      else node.targets):
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return frozenset(out)


# ================================================== 5. RETRIEVAL (INDEXED)
class TemplateStore:
    """Shard + posting-list index over the mined catalog.

    NOT a scan. `retrieve` touches one shard bucket and the postings for the
    template's required names; a target whose shard has no entries costs one
    dict lookup regardless of catalog size. `stats()["scanned"]` is asserted in
    the tests so a future refactor cannot quietly turn this back into a loop.
    """

    def __init__(self, templates=()):
        self.templates = tuple(templates)
        self.by_tid = {t.tid: t for t in self.templates}
        self.shards = {}
        self.by_required_name = {}
        self._scanned = 0
        for t in self.templates:
            for key in self._shard_keys(t):
                self.shards.setdefault(key, []).append(t.tid)
            for nm in t.precondition.get("requires_names", ()):
                self.by_required_name.setdefault(nm, set()).add(t.tid)
        for key in self.shards:
            self.shards[key] = tuple(sorted(set(self.shards[key])))

    # ------------------------------------------------------------- shards
    @staticmethod
    def _shard_keys(t):
        """Index keys a template answers to.

        Keyed on the CONTROL/DATAFLOW role first (the design's retrieval order
        #1 and #2), never on symptom words: a shared noun with a different
        owner must not retrieve a template.
        """
        pre = t.precondition
        loop = pre.get("loop_bucket", "flat")
        exit_kind = t.effect.get("exit_kind", "none")
        # DELIBERATELY NOT keyed on the donor's statement-list role. A guard
        # mined at a function's top level is the same guard inside a `try` or a
        # `with`; keying on it made every template retrievable ONLY at its own
        # mining context, which measured as 0 reachable on the v3 board with
        # the gold's own template planted in the store. `list_role` survives as
        # a rerank feature and, when `list_role_strict`, as a hard filter in
        # `compatible` — it is not an index key.
        return [(t.role, loop, exit_kind)]

    @staticmethod
    def point_shard_keys(role, point):
        """Which shard buckets a target insertion point may draw from.

        A `flat` template is legal inside a loop (its effects do not touch the
        back edge); a `loop` template is NOT legal outside one, which is the
        asymmetry that keeps a `continue` from being inserted into a flat body.
        """
        loops = ["flat"] if point.loop_bucket == "flat" else ["flat", "loop"]
        exits = ["none", "return", "raise"]
        if point.loop_bucket == "loop":
            exits = exits + ["continue", "break"]
        out = []
        for lb in loops:
            for ek in exits:
                out.append((role, lb, ek))
        return out

    # ----------------------------------------------------------- retrieval
    def retrieve(self, facts, point, roles=ROLES,
                 limit=MAX_TEMPLATES_PER_POINT):
        tids = []
        for role in roles:
            for key in self.point_shard_keys(role, point):
                tids.extend(self.shards.get(key, ()))
        seen, out = set(), []
        for tid in tids:
            if tid in seen:
                continue
            seen.add(tid)
            t = self.by_tid[tid]
            self._scanned += 1
            need = t.precondition.get("requires_names", ())
            if any(not facts.name_available(n, point) for n in need):
                continue
            if not compatible(t, facts, point):
                continue
            out.append(t)
        out.sort(key=lambda t: (ROLE_PRIORITY[t.role], -t.n_instances,
                                len(t.holes), t.n_lines, t.tid))
        return out[:limit]

    def stats(self):
        return {"templates": len(self.templates), "shards": len(self.shards),
                "scanned": self._scanned}

    # ------------------------------------------------------------ storage
    def to_json(self, coverage=None):
        return {"schema": SCHEMA_VERSION,
                "coverage": dict(coverage or {}),
                "templates": [t.to_json() for t in self.templates]}

    def save(self, path=STORE_PATH, coverage=None):
        with open(path, "w") as fh:
            json.dump(self.to_json(coverage), fh, indent=2, sort_keys=True)
            fh.write("\n")

    @staticmethod
    def load(path=STORE_PATH):
        if not os.path.exists(path):
            return TemplateStore(())
        with open(path) as fh:
            blob = json.load(fh)
        if blob.get("schema") != SCHEMA_VERSION:
            # A stale catalog is refused, never half-read: a template whose
            # signature vocabulary moved would unify against the wrong facts.
            return TemplateStore(())
        return TemplateStore([BlockTemplate.from_json(d)
                              for d in blob.get("templates", [])])


# ================================================== 6. HARD COMPATIBILITY
def compatible(template, facts, point):
    """Dataflow/CFG hard filter (`a_block_synthesis.md`: NOT a reranker).

    Returns True only when the template's own control requirements hold at this
    point. API overlap and symptom nouns are deliberately absent — they are
    reranking and hint signals, and letting either admit a candidate is exactly
    the "shared noun, different owner" failure the design forbids.
    """
    pre = template.precondition
    eff = template.effect
    if pre.get("requires_loop") and point.loop_bucket != "loop":
        return False
    if eff.get("exit_kind") in ("continue", "break") \
            and point.loop_bucket != "loop":
        return False
    if pre.get("requires_async") and not facts.is_async:
        return False
    if pre.get("requires_handler") and not point.in_handler:
        return False
    if point.list_role not in pre.get("list_roles", ["fn_body"]) \
            and pre.get("list_role_strict", False):
        return False
    # Only calls still LITERAL in the skeleton are a hard requirement; a holed
    # `Call<RegisteredAPI>` is checked when it is bound, against the target's
    # own resolvable APIs (`_domain_for`).
    for call in eff.get("frozen_calls", ()):
        if not facts.call_available(call):
            return False
    return True


# ===================================================== 7. HOLE UNIFICATION
def _domain_for(hole, facts, point):
    """Deterministic, bounded binding domain for one hole."""
    if hole.kind == "Name<Input>":
        return list(facts.params)
    if hole.kind == "Name<BoundLocal>":
        return list(facts.locals_live_at(point))
    if hole.kind in ("Name<Receiver>", "Name<Free>"):
        return list(facts.params) + list(facts.locals_live_at(point))
    if hole.kind == "Call<RegisteredAPI>":
        # The TARGET's resolvable APIs only. A donor's call name is preferred
        # when the target also has it, but a call the target cannot resolve is
        # never bound — that is the difference between reusing a structure and
        # inventing an API.
        dom = sorted(c for c in facts.calls if facts.call_available(c))
        if hole.default and facts.call_available(hole.default):
            dom = [hole.default] + [c for c in dom if c != hole.default]
        return dom
    if hole.kind == "Literal<const>":
        return [hole.default]
    return []


def unify(template, facts, point, oracle_fills=None,
          max_candidates=MAX_BINDING_CANDIDATES):
    """Bind every hole against the target symbol table.

    Returns a deterministic list of fill dicts (possibly empty). ``oracle_fills``
    is the offline instrument's channel — the same convention as
    `holes.oracle_fill`: it reads gold, it is NEVER available on a fix path,
    and a number produced with it is an UPPER BOUND, not a result.
    """
    if not compatible(template, facts, point):
        return []
    domains = []
    for h in template.holes:
        if oracle_fills and h.placeholder in oracle_fills:
            domains.append([oracle_fills[h.placeholder]])
            continue
        dom = _domain_for(h, facts, point)
        # Prefer the mined name when the target still has it: a donor block
        # written against `payload` should bind `payload` when the target has
        # one, rather than the alphabetically-first parameter.
        dom = sorted(set(dom), key=lambda v: (v != h.default, repr(v)))
        if not dom:
            return []
        domains.append(dom[:max_candidates])
    fills = [{}]
    for h, dom in zip(template.holes, domains):
        nxt = []
        for base in fills:
            for v in dom:
                if len(nxt) >= max_candidates:
                    break
                d = dict(base)
                d[h.placeholder] = v
                nxt.append(d)
        fills = nxt
    return fills[:max_candidates]


_ALPHA_LOCAL_RE = re.compile(r"\b_b(\d+)\b")


def alpha_locals(template):
    """The alpha-renamed block locals (`_b0`, `_b1`, ...) in definition order."""
    return sorted(set(_ALPHA_LOCAL_RE.findall(template.skeleton)), key=int)


def fresh_names(template, facts, point):
    """Collision-free concrete names for the template's own block locals.

    A template's `_b0` is a local the block DEFINES. Emitting it verbatim would
    silently shadow a target local of the same name, so the name is only kept
    when the target does not already carry it.
    """
    taken = set(facts.params) | set(facts.assign_line) | set(facts.module_names)
    out = {}
    for idx in alpha_locals(template):
        base = "_b%s" % idx
        name = base
        n = 0
        while name in taken:
            n += 1
            name = "%s_%d" % (base, n)
        taken.add(name)
        out[base] = name
    return out


def render(template, fills, indent, renames=None):
    """Skeleton + fills -> concrete block text at ``indent`` columns, or None.

    ``renames`` concretises the alpha-renamed block locals. It is a SEPARATE
    channel from ``fills`` on purpose: a block local is not a hole, it never
    binds to a target value, and letting it share the fill dict would let a
    donor local silently capture a target name.
    """
    body = template.skeleton
    for old, new in sorted((renames or {}).items(), key=lambda kv: -len(kv[0])):
        body = _ALPHA_LOCAL_RE.sub(
            lambda m, o=old, n=new: n if ("_b" + m.group(1)) == o
            else m.group(0), body)
    for h in template.holes:
        if h.placeholder not in fills:
            return None
        val = fills[h.placeholder]
        # A NAME hole substitutes an identifier verbatim; a LITERAL hole must
        # be re-quoted or a mined `"draft"` would render as a bare name and
        # silently become a NameError at runtime rather than a parse failure
        # the gate could catch.
        if h.kind == "Literal<const>":
            text_val = repr(val)
        else:
            text_val = val if isinstance(val, str) else repr(val)
        body = body.replace(h.placeholder, text_val)
    if "__BH" in body:
        return None                          # an unfilled hole is never shipped
    if parse_block(body) is None:
        return None
    pad = " " * indent
    return "\n".join((pad + ln) if ln.strip() else "" for ln in
                     body.rstrip("\n").split("\n"))


# ============================================== 8. CANDIDATE CONSTRUCTION
def make_insertion(span_lines, point, block_text, template, fills):
    """One atomic block insertion as an `edit_sets.EditSet`.

    The insertion is expressed as a REPLACEMENT of the anchor line by
    `block + anchor`, which is what makes it a single `apply_candidate` splice
    and therefore atomic: there is no state in which half the block landed.
    """
    li = point.line
    if not (0 <= li < len(span_lines)):
        return None
    anchor = span_lines[li]
    new_block = block_text + "\n" + anchor
    edit = es.SpanEdit(anchor, new_block, role="block_insert")
    try:
        return es.EditSet((li, li), anchor, new_block, [edit],
                          "block_template:%s" % template.tid,
                          sorted(str(v) for v in fills.values()))
    except Exception:
        return None


# =========================================================== 9. THE SEAM
#: THE ONE FUNCTION THE FIXER MAY CALL.
#:
#: Wiring is the NEXT lane's work and is deliberately not done here. When it is
#: done, it is these three lines and nothing else:
#:
#:   1. fixer.py FAMILY_MAP (the only grant path):
#:          "block_template": "block_template",
#:   2. gym_config.py fix.families default list: append "block_template".
#:   3. the enumerator arm, next to the existing family calls:
#:          import blocks
#:          cands = blocks.block_insertion_candidates(span, start, file_lines)
#:      Every returned candidate is already an `edit_sets.EditSet`, so the
#:      existing `edit_sets.apply_candidate(lines, mut)` site applies it
#:      unchanged and `None` still means skip/fail-closed.
#:
#: `oracle_fills` exists for `probes/block_miner.py --reach` ONLY. A fix path
#: passing it would be reporting an upper bound as a result.
BLOCK_FIXER_SEAM = (
    "blocks.block_insertion_candidates(span_lines, span_start, file_lines, "
    "store=None, limit=BLOCK_CANDIDATE_LIMIT, oracle_fills=None, "
    "fn_name=None) -> list[edit_sets.EditSet]")


def block_insertion_candidates(span_lines, span_start, file_lines=None,
                               store=None, limit=BLOCK_CANDIDATE_LIMIT,
                               oracle_fills=None, fn_name=None,
                               roles=ROLES):
    """THE SEAM. Bounded, deterministic block-insertion candidates for a span.

    ``span_lines`` is the target function's source lines (the same object every
    other family in this tree receives); ``span_start`` is its offset into
    ``file_lines``, used only to read module-level names for API resolution.
    """
    src = textwrap.dedent("\n".join(span_lines))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    fn_node = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and (fn_name is None or node.name == fn_name):
            fn_node = node
            break
    if fn_node is None:
        return []
    mod_names = module_level_names("\n".join(file_lines)) if file_lines \
        else frozenset()
    facts = TargetFacts(fn_node, mod_names)
    store = store if store is not None else TemplateStore.load()

    # Indentation is measured on the RAW span, not the dedented parse, so the
    # rendered block lands at the file's real column.
    dedent_shift = _dedent_shift(span_lines)
    out = []
    for point in facts.points:
        for t in store.retrieve(facts, point, roles=roles):
            renames = fresh_names(t, facts, point)
            for fills in unify(t, facts, point, oracle_fills):
                text = render(t, fills, point.indent + dedent_shift, renames)
                if text is None:
                    continue
                cand = make_insertion(span_lines, point, text, t, fills)
                if cand is not None:
                    out.append(cand)
                if len(out) >= limit:
                    return out
    return out


def _dedent_shift(span_lines):
    for ln in span_lines:
        if ln.strip():
            return len(ln) - len(ln.lstrip())
    return 0
