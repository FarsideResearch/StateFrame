"""Tree-sitter multi-language extractors for the code call graph.

Sibling of ``call_graph.py``'s stdlib-``ast`` Python path. This module produces
the EXACT SAME per-file extraction shape (``SymbolDef`` / ``CallSite`` /
import table, wrapped in ``_FileExtract``) for the languages our stack actually
uses — TypeScript, JavaScript, and Vue SFCs — plus a few cheap tree-sitter
adds (Go, Rust, Java). ``call_graph.py`` owns the resolution/graph/query layer;
this module only turns source text into defs + call sites + a local import
table, so every downstream feature (explore, callers, callees, impact_radius,
signature skeletons, test-impact) works uniformly across languages.

The Python extractor stays in ``call_graph.py`` on stdlib ``ast`` (proven, exact
line numbers, zero parser-availability risk) — this module never touches ``.py``.

Design rules (inherited from call_graph.py — rule #158):
  * BIAS TO OVER-REPORT. A call whose target can't be pinned down is emitted as
    a bare short-name (call_graph's resolver then links it to every same-named
    def) rather than dropped. A missed caller is the silent break we exist to
    catch; a too-wide caller set is safe to review.
  * NEVER RAISE. A parse failure, a missing grammar, or a tree-sitter version
    mismatch degrades to an empty extract + a debug log. Callers get []; the
    Python graph is unaffected.
  * BOUNDARY HONESTY. A runtime-resolved call (``handlers[k]()``, dynamic
    ``import()``, ``emitter.emit('evt')``) is NOT faked into a static edge — it
    is announced as a ``DispatchBoundary`` (see call_graph.dispatch_boundaries),
    never emitted as a wrong call edge.

Extractor config model + node-type lists ported from CodeGraph
(colbymchenry/codegraph, MIT — declarative ``LanguageExtractor`` in
``src/extraction/languages/*.ts`` and the Vue ``<script>``-block strategy in
``src/extraction/vue-extractor.ts``). Reimplemented in Python against the
``tree_sitter_language_pack`` grammars actually shipped in the agent-runner
image (tree-sitter 0.26 core), so it's a clean re-implementation, not a copy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tree-sitter availability probe (mirrors repo_intel/signatures.py) ─────────
# The agent-runner image pins tree-sitter + tree_sitter_language_pack; a coding
# task can nonetheless land in an env where the core/grammar versions disagree,
# so we probe once and degrade to the Python-only graph rather than raise.

try:  # pragma: no cover - import guard exercised only in a pack-less env
    import tree_sitter_language_pack as _tslp  # type: ignore[import]
    _TSLP_IMPORTED = True
except Exception as _exc:  # noqa: BLE001 - any import problem => degrade
    _tslp = None  # type: ignore[assignment]
    _TSLP_IMPORTED = False
    logger.warning(
        "ts_extract: tree_sitter_language_pack unavailable (%s) — TS/JS/Vue/Go/"
        "Rust/Java call-graph extraction disabled; Python graph unaffected.",
        _exc,
    )

_tslp_functional: Optional[bool] = None


def is_tree_sitter_functional() -> bool:
    """Whether the pack can actually parse (not just import). Cached after first probe.

    A tree-sitter core bump can break a grammar binding's API at parse time even
    when the import succeeds — we prove a real parse before advertising support,
    exactly like ``repo_intel.signatures.is_tree_sitter_functional``.

    The probe must exercise the SAME Node API surface ``_TSVisitor._walk`` and
    ``_build_import_table`` actually use — ``root_node``, ``.children``,
    ``.type`` — not just ``parser.parse(...)``. A pack/core version skew (e.g.
    ``tree-sitter-language-pack`` 1.x + ``tree-sitter`` 0.25, whose Node has no
    ``.type``/``.children`` property, ``root_node`` as a method not a property,
    ``.kind``/``.child()`` instead) still returns a tree from ``parse()`` — a
    probe that stops there reports "functional" while every real extraction
    AttributeErrors inside the walk (rule #158/#189-class: looks parsed,
    is actually broken).
    """
    global _tslp_functional
    if _tslp_functional is not None:
        return _tslp_functional
    if not _TSLP_IMPORTED:
        _tslp_functional = False
        return False
    try:
        parser = _tslp.get_parser("typescript")
        tree = _parse(parser, b"const x = 1; function f() { g(); }")
        root = tree.root_node  # property access, not a method call
        _ = root.type
        children = list(root.children)
        if not children:
            raise RuntimeError("root_node.children is empty for a non-empty parse")
        # Touch child_by_field_name — the primary way defs/calls are located —
        # so a binding that has `.children` but a broken/missing field API
        # (another plausible skew shape) is also caught here.
        _ = children[0].child_by_field_name("name")
        _tslp_functional = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ts_extract: tree_sitter_language_pack installed but non-functional "
            "(%s) — multi-language call graph disabled until the pin is reconciled.",
            exc,
        )
        _tslp_functional = False
    return _tslp_functional


def _parse(parser, data: bytes):
    """Parse bytes, tolerating both the bytes-API and the older str-API packs."""
    try:
        return parser.parse(data)
    except TypeError:  # a pack build that wants str
        return parser.parse(data.decode("utf-8", errors="replace"))


# ── Extension → grammar map ──────────────────────────────────────────────────
# .vue is handled specially (script-block extraction), so it is NOT in this map.
_LANG_FOR_EXT: dict[str, str] = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",  # the js grammar parses jsx in the pack
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

# Extensions this module claims (call_graph dispatches these away from the ast path).
SUPPORTED_EXTENSIONS = frozenset(set(_LANG_FOR_EXT) | {".vue"})


# ── Declarative per-language extractor config (ported from CodeGraph) ─────────

@dataclass(frozen=True)
class LangSpec:
    """Node-type sets that describe one language's shape to the generic walker.

    Ported from CodeGraph's ``LanguageExtractor`` (MIT). Only the fields our
    call-graph model needs are kept — we emit call/def edges, not the full
    node taxonomy (extends/implements/enum-member) CodeGraph persists.
    """
    function_types: frozenset[str]
    class_types: frozenset[str]
    method_types: frozenset[str]
    interface_types: frozenset[str]
    struct_types: frozenset[str]
    type_alias_types: frozenset[str]
    call_types: frozenset[str]
    import_types: frozenset[str]
    variable_types: frozenset[str]
    name_field: str = "name"
    # The child node types that carry a callable value when assigned to a
    # variable declarator (``const f = () => …`` / ``const f = function …``) —
    # so a named arrow/function-expression const becomes a def.
    callable_value_types: frozenset[str] = frozenset()
    # Field name on a call node holding the callee expression.
    call_func_field: str = "function"
    # Loop constructs — drive max_loop_depth / calls_in_loops. Deliberately a
    # generic node-type set, not a per-language analyser (see _node_properties).
    loop_types: frozenset[str] = frozenset()
    # Branch points — each adds 1 to cyclomatic complexity.
    branch_types: frozenset[str] = frozenset()


# Loop / branch node types. ONE union per family rather than a hand-tuned set
# per language: the node names are already disjoint across grammars, so a shared
# set costs nothing and cannot drift out of step with a language we add later.
# These metrics are HINTS — over-inclusion is cheap, a missing signal is not.
_C_LIKE_LOOPS = frozenset({
    "for_statement", "for_in_statement", "for_of_statement", "while_statement",
    "do_statement", "enhanced_for_statement", "range_clause",
    "for_expression", "while_expression", "loop_expression",
})
_C_LIKE_BRANCHES = frozenset({
    "if_statement", "if_expression", "ternary_expression", "conditional_expression",
    "catch_clause", "switch_case", "case_clause", "expression_case",
    "default_case", "match_arm", "type_case", "communication_case",
}) | _C_LIKE_LOOPS

# --- TypeScript / TSX (also drives Vue <script lang="ts">) -------------------
_TS_SPEC = LangSpec(
    function_types=frozenset({"function_declaration"}),
    class_types=frozenset({"class_declaration", "abstract_class_declaration"}),
    method_types=frozenset({"method_definition", "public_field_definition"}),
    interface_types=frozenset({"interface_declaration"}),
    struct_types=frozenset(),
    type_alias_types=frozenset({"type_alias_declaration"}),
    call_types=frozenset({"call_expression", "new_expression"}),
    import_types=frozenset({"import_statement"}),
    variable_types=frozenset({"lexical_declaration", "variable_declaration"}),
    callable_value_types=frozenset({"arrow_function", "function_expression"}),
    loop_types=_C_LIKE_LOOPS,
    branch_types=_C_LIKE_BRANCHES,
)

# --- JavaScript / JSX --------------------------------------------------------
_JS_SPEC = LangSpec(
    function_types=frozenset({"function_declaration"}),
    class_types=frozenset({"class_declaration"}),
    method_types=frozenset({"method_definition", "field_definition"}),
    interface_types=frozenset(),
    struct_types=frozenset(),
    type_alias_types=frozenset(),
    call_types=frozenset({"call_expression", "new_expression"}),
    import_types=frozenset({"import_statement"}),
    variable_types=frozenset({"lexical_declaration", "variable_declaration"}),
    callable_value_types=frozenset({"arrow_function", "function_expression"}),
    loop_types=_C_LIKE_LOOPS,
    branch_types=_C_LIKE_BRANCHES,
)

# --- Go ----------------------------------------------------------------------
_GO_SPEC = LangSpec(
    function_types=frozenset({"function_declaration"}),
    class_types=frozenset(),
    method_types=frozenset({"method_declaration"}),
    interface_types=frozenset(),
    struct_types=frozenset(),
    type_alias_types=frozenset({"type_spec"}),
    call_types=frozenset({"call_expression"}),
    import_types=frozenset({"import_declaration"}),
    variable_types=frozenset({"var_declaration", "short_var_declaration", "const_declaration"}),
    callable_value_types=frozenset(),
    loop_types=_C_LIKE_LOOPS,
    branch_types=_C_LIKE_BRANCHES,
)

# --- Rust --------------------------------------------------------------------
_RUST_SPEC = LangSpec(
    function_types=frozenset({"function_item"}),
    class_types=frozenset(),
    method_types=frozenset({"function_item"}),  # inside impl_item
    interface_types=frozenset({"trait_item"}),
    struct_types=frozenset({"struct_item"}),
    type_alias_types=frozenset({"type_item"}),
    call_types=frozenset({"call_expression"}),
    import_types=frozenset({"use_declaration"}),
    variable_types=frozenset({"let_declaration", "const_item", "static_item"}),
    callable_value_types=frozenset(),
    loop_types=_C_LIKE_LOOPS,
    branch_types=_C_LIKE_BRANCHES,
)

# --- Java --------------------------------------------------------------------
_JAVA_SPEC = LangSpec(
    function_types=frozenset(),
    class_types=frozenset({"class_declaration"}),
    method_types=frozenset({"method_declaration", "constructor_declaration"}),
    interface_types=frozenset({"interface_declaration"}),
    struct_types=frozenset(),
    type_alias_types=frozenset(),
    call_types=frozenset({"method_invocation", "object_creation_expression"}),
    import_types=frozenset({"import_declaration"}),
    variable_types=frozenset({"local_variable_declaration"}),
    callable_value_types=frozenset(),
    call_func_field="name",  # java method_invocation carries a `name` field, not `function`
    loop_types=_C_LIKE_LOOPS,
    branch_types=_C_LIKE_BRANCHES,
)

_SPEC_FOR_LANG: dict[str, LangSpec] = {
    "typescript": _TS_SPEC,
    "tsx": _TS_SPEC,
    "javascript": _JS_SPEC,
    "go": _GO_SPEC,
    "rust": _RUST_SPEC,
    "java": _JAVA_SPEC,
}

# ── Entrypoint heuristics (parallel to call_graph._is_entrypoint) ────────────
# A def with these markers is externally-invoked, so it must not be flagged a
# dead-code orphan just because nothing local calls it.
_ENTRYPOINT_NAME_RE = re.compile(r"^(?:test_|Test|main$|setup$|teardown$)")
# Framework decorators/annotations (TS `@Get()`, Java `@RestController`) that
# mark a symbol as an entrypoint. Short-name match, over-inclusive by design.
_ENTRYPOINT_DECORATOR_NAMES = frozenset({
    "Get", "Post", "Put", "Patch", "Delete", "Route", "Controller",
    "RestController", "RequestMapping", "GetMapping", "PostMapping",
    "Component", "Injectable", "Test", "Bean", "EventListener",
})

# Method names invoked implicitly (constructors / lifecycle) — never orphans.
_LIFECYCLE_METHOD_NAMES = frozenset({
    "constructor", "render", "setup", "created", "mounted", "beforeMount",
    "beforeDestroy", "unmounted", "toString", "equals", "hashCode",
})


def _is_entrypoint_name(name: str, kind: str, decorator_names: set[str]) -> bool:
    if _ENTRYPOINT_NAME_RE.search(name):
        return True
    if kind == "class":
        return True  # classes/components are instantiated externally
    if name in _LIFECYCLE_METHOD_NAMES:
        return True
    if decorator_names & _ENTRYPOINT_DECORATOR_NAMES:
        return True
    return False


# ── Signature extraction ─────────────────────────────────────────────────────
# call_graph.Signature is Python-centric (posonly/kwonly/*args). Cross-language
# callables don't map cleanly onto that, and the T29 breaking-change diff runs
# on the Python `ast` path. For TS/JS/etc. we record a coarse positional arity
# (param count + whether a rest param makes it unbounded) into the same
# Signature so signature skeletons and a same-language arity diff still work.


def _import_signature():
    # Local import to avoid a hard module-load dependency cycle (call_graph
    # imports nothing from here at module load; we reach back only when used).
    from .call_graph import Signature  # noqa: PLC0415
    return Signature


def _param_signature(param_count: int, has_rest: bool):
    Signature = _import_signature()
    return Signature(
        args=tuple(f"a{i}" for i in range(param_count)),
        has_star_args=has_rest,
    )


# ── The generic tree-sitter walk ─────────────────────────────────────────────


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_of(source: bytes, node, spec: LangSpec) -> Optional[str]:
    """Best-effort declared name of a def node via its name field, else first
    identifier-ish child."""
    n = node.child_by_field_name(spec.name_field)
    if n is not None:
        return _node_text(source, n)
    for c in node.children:
        if c.type in ("identifier", "type_identifier", "property_identifier",
                       "field_identifier", "name"):
            return _node_text(source, c)
    return None


def _decorator_names(source: bytes, node) -> set[str]:
    """Short names of decorators/annotations attached above a def node.

    TS decorators are ``decorator`` siblings preceding the def inside its parent;
    Java annotations are ``marker_annotation`` / ``annotation`` in the modifiers.
    We scan the node's own subtree shallowly plus preceding siblings.
    """
    names: set[str] = set()

    def _scan(n, depth=0):
        if depth > 2:
            return
        for c in n.children:
            if c.type in ("decorator", "annotation", "marker_annotation"):
                # last identifier component of the decorator/annotation
                ident = None
                for g in c.children:
                    if g.type in ("identifier", "type_identifier"):
                        ident = _node_text(source, g)
                    elif g.type == "call_expression":
                        f = g.child_by_field_name("function")
                        if f is not None:
                            ident = _node_text(source, f).split(".")[-1]
                if ident:
                    names.add(ident.split(".")[-1])
        # modifiers block (java) holds annotations as children too
        for c in n.children:
            if c.type == "modifiers":
                _scan(c, depth + 1)

    _scan(node)
    return names


def _param_facts(node) -> tuple[int, bool]:
    """(positional param count, has_rest) from a def node's parameter list."""
    for c in node.children:
        if c.type in ("formal_parameters", "parameters", "parameter_list",
                      "argument_list"):
            count = 0
            has_rest = False
            for p in c.children:
                if p.type in ("required_parameter", "optional_parameter",
                              "parameter", "identifier", "typed_parameter",
                              "spread_parameter"):
                    count += 1
                if p.type in ("rest_pattern", "spread_parameter", "variadic_parameter"):
                    has_rest = True
            return count, has_rest
    return 0, False


def _node_properties(node, spec: LangSpec, source: bytes, short_name: str) -> dict:
    """Static complexity metrics for one tree-sitter def node. Never raises.

    The SAME shape call_graph._complexity_properties produces for Python, so
    every consumer (hotspots, explore, find_definition) reads one set of keys
    regardless of language. Computed generically from the LangSpec node-type
    sets — deliberately NOT a per-language analyser, which is how this kind of
    thing turns into an unmaintainable monster. Nested function/class bodies are
    skipped: each owns its own def and therefore its own metrics.
    """
    from .call_graph import _empty_properties  # noqa: PLC0415

    nested = (spec.function_types | spec.method_types | spec.class_types
              | spec.callable_value_types)
    length = max(1, (node.end_point[0] - node.start_point[0]) + 1)
    props = _empty_properties(length)
    # Mirrors the Python path: a conditional early exit is a base case too, so a
    # correctly-written recursion is not flagged as unguarded.
    state = {"conditional_exit": False, "bare_self_call": False}

    def _walk(n, loop_depth: int, guard_depth: int, depth: int) -> None:
        if depth > 200:  # pathological nesting — bail rather than blow the stack
            return
        for child in n.children:
            t = child.type
            if t in nested:
                continue
            if t in spec.branch_types:
                props["cyclomatic"] += 1
            is_loop = t in spec.loop_types
            child_loop_depth = loop_depth + 1 if is_loop else loop_depth
            if is_loop and child_loop_depth > props["max_loop_depth"]:
                props["max_loop_depth"] = child_loop_depth
            if t in ("return_statement", "throw_statement") and guard_depth > 0:
                state["conditional_exit"] = True
            if t in spec.call_types:
                if loop_depth >= 1:
                    props["calls_in_loops"] += 1
                func = child.child_by_field_name(spec.call_func_field)
                if func is not None and _node_text(source, func).split(".")[-1] == short_name:
                    props["recursive"] = True
                    if guard_depth == 0:
                        state["bare_self_call"] = True
            _walk(child, child_loop_depth,
                  guard_depth + (1 if t in spec.branch_types else 0), depth + 1)

    try:
        _walk(node, 0, 0, 0)
    except RecursionError:  # pragma: no cover - very deep trees
        logger.debug("ts_extract: recursion limit computing properties for %s", short_name)
    props["unguarded_recursion"] = bool(
        state["bare_self_call"] and not state["conditional_exit"])
    props["linear_scan_in_loop"] = (
        props["calls_in_loops"] > 0 or props["max_loop_depth"] >= 2
    )
    return props


# Constructor-call node types → the field holding the callee/class-type
# expression. `call_types` includes these per-language (TS/JS `new_expression`,
# Java `object_creation_expression`); Rust has no separate constructor-call
# syntax (`Foo::new(...)` is already a plain `call_expression`, and struct
# literals `Foo { .. }` are not calls at all), so it is deliberately absent.
_CONSTRUCTOR_CALL_FIELD: dict[str, str] = {
    "new_expression": "constructor",
    "object_creation_expression": "type",
}


class _TSVisitor:
    """Single-pass recursive walk collecting defs + call sites with scope.

    Mirrors ``call_graph._FileVisitor`` semantics: a symbol id is
    ``<relpath>::<QualName>``, a method is a def whose nearest enclosing scope
    is a class/impl body, and every call site records the enclosing def as its
    caller (module top-level calls attribute to a synthetic ``<module>`` node).
    """

    def __init__(self, relpath: str, source: bytes, spec: LangSpec,
                 import_table: dict[str, str], line_offset: int = 0,
                 qual_prefix: str = ""):
        self.relpath = relpath
        self.source = source
        self.spec = spec
        self.import_table = import_table
        self.line_offset = line_offset      # for Vue: <script> block start line
        self.qual_prefix = qual_prefix      # for Vue: "<Component>." on script defs
        self.defs: list = []
        self.calls: list = []
        self._scope: list[str] = []
        self._class_depth = 0
        self._func_depth = 0

    # ---- id helpers ----
    def _qual(self, name: str) -> str:
        inner = ".".join(self._scope + [name]) if self._scope else name
        return f"{self.qual_prefix}{inner}" if self.qual_prefix else inner

    def _sym(self, qualname: str) -> str:
        return f"{self.relpath}::{qualname}"

    def _current_caller(self) -> str:
        if self._scope:
            inner = ".".join(self._scope)
            q = f"{self.qual_prefix}{inner}" if self.qual_prefix else inner
            return self._sym(q)
        # module top-level (or Vue component setup body) — a synthetic node.
        base = self.qual_prefix.rstrip(".") if self.qual_prefix else "<module>"
        return self._sym(base or "<module>")

    def _line(self, node) -> int:
        return node.start_point[0] + 1 + self.line_offset

    def _end_line(self, node) -> int:
        return node.end_point[0] + 1 + self.line_offset

    # ---- def emission ----
    def _emit_def(self, node, name: str, kind: str) -> None:
        from .call_graph import SymbolDef  # noqa: PLC0415
        qual = self._qual(name)
        pc, rest = _param_facts(node)
        decs = _decorator_names(self.source, node)
        self.defs.append(SymbolDef(
            symbol=self._sym(qual), file=self.relpath, qualname=qual,
            short_name=name, lineno=self._line(node), end_lineno=self._end_line(node),
            kind=kind,
            is_entrypoint=_is_entrypoint_name(name, kind, decs),
            signature=_param_signature(pc, rest),
            # Classes/interfaces/types are containers, not bodies of work — the
            # metrics are meaningless there and an empty dict says so honestly.
            properties=(
                _node_properties(node, self.spec, self.source, name)
                if kind in ("function", "method") else {}
            ),
        ))

    # ---- call emission ----
    def _emit_call(self, node) -> None:
        from .call_graph import CallSite  # noqa: PLC0415
        # `new Foo()` (TS/JS `new_expression`) and Java's `new Foo()`
        # (`object_creation_expression`) carry the callee under a DIFFERENT
        # field than a plain call — `constructor` / `type` respectively, never
        # `spec.call_func_field` ("function"/"name"). Without this, a
        # constructor invocation is invisible to the call graph entirely
        # (rule #158-class: looks parsed, produces zero edges).
        field = _CONSTRUCTOR_CALL_FIELD.get(node.type, self.spec.call_func_field)
        func = node.child_by_field_name(field)
        if func is None:
            # java method_invocation without the field (qualified) — fall back
            for c in node.children:
                if c.type in ("identifier", "field_access", "member_expression"):
                    func = c
                    break
        if func is None:
            return
        short, hint_self, hint_module = self._resolve_call_target(func)
        if short is None:
            return  # dynamic/computed — a DispatchBoundary announces it, not an edge
        args = node.child_by_field_name("arguments")
        n_pos = 0
        has_star = False
        if args is not None:
            for a in args.children:
                if a.type in ("spread_element", "spread_argument"):
                    has_star = True
                elif a.is_named:
                    n_pos += 1
        self.calls.append(CallSite(
            caller=self._current_caller(),
            callee_short=short,
            n_positional=n_pos,
            has_star=has_star,
            keywords=(),
            hint_self=hint_self,
            hint_module=hint_module,
            lineno=self._line(node),
        ))

    def _resolve_call_target(self, func) -> tuple[Optional[str], bool, Optional[str]]:
        """(callee_short, is_self_call, module_hint). Returns (None,..) for a
        runtime-resolved target (computed subscript / dynamic import) so we emit
        NO fake edge — the dispatch-boundary scan announces those honestly."""
        ft = func.type
        if ft in ("identifier", "type_identifier"):
            return _node_text(self.source, func), False, None
        if ft == "generic_type":  # Java `new Foo<Bar>()` — unwrap to the raw type name
            for c in func.children:
                if c.type in ("type_identifier", "scoped_type_identifier"):
                    return _node_text(self.source, c).split(".")[-1], False, None
            return None, False, None
        if ft in ("member_expression", "field_access"):
            prop = func.child_by_field_name("property") or func.child_by_field_name("field")
            base = func.child_by_field_name("object") or (
                func.children[0] if func.children else None)
            short = _node_text(self.source, prop) if prop is not None else None
            if short is None:
                # last identifier child
                for c in reversed(func.children):
                    if c.type in ("property_identifier", "identifier", "field_identifier"):
                        short = _node_text(self.source, c)
                        break
            if short is None:
                return None, False, None
            if base is not None and base.type in ("this", "super"):
                return short, True, None
            if base is not None and base.type == "identifier":
                return short, False, _node_text(self.source, base)
            return short, False, None
        if ft in ("scoped_identifier",):  # java/rust Foo::bar, a.b.c
            for c in reversed(func.children):
                if c.type in ("identifier", "field_identifier"):
                    return _node_text(self.source, c), False, None
        # subscript_expression, `import`, parenthesized dynamic expr, etc. → boundary
        return None, False, None

    # ---- variable-declarator with a callable value (const f = () => …) ----
    def _maybe_callable_const(self, decl_node) -> None:
        """Emit a def for `const name = <arrow|fn-expr>` and walk its body scoped."""
        for vd in decl_node.children:
            if vd.type != "variable_declarator":
                continue
            name_n = vd.child_by_field_name("name")
            value_n = vd.child_by_field_name("value")
            if value_n is None:
                # value may be an unnamed positional child
                kids = [c for c in vd.children if c.is_named]
                value_n = kids[-1] if len(kids) >= 2 else None
            if name_n is None or value_n is None:
                self._walk(vd)
                continue
            if value_n.type in self.spec.callable_value_types:
                name = _node_text(self.source, name_n)
                is_method = self._class_depth > 0 and self._func_depth == 0
                self._emit_def(value_n, name, "method" if is_method else "function")
                self._scope.append(name)
                self._func_depth += 1
                self._walk(value_n)
                self._func_depth -= 1
                self._scope.pop()
            else:
                self._walk(vd)

    # ---- the recursive walk ----
    def _walk(self, node) -> None:
        spec = self.spec
        t = node.type

        if t in spec.call_types:
            self._emit_call(node)
            for c in node.children:
                self._walk(c)
            return

        if t in spec.variable_types and spec.callable_value_types:
            self._maybe_callable_const(node)
            return

        if t in spec.class_types or t == "impl_item":
            name = _name_of(self.source, node, spec) or "<anon>"
            if t in spec.class_types:
                self._emit_def(node, name, "class")
            self._scope.append(name)
            self._class_depth += 1
            for c in node.children:
                self._walk(c)
            self._class_depth -= 1
            self._scope.pop()
            return

        if t in spec.interface_types or t in spec.struct_types or t in spec.type_alias_types:
            name = _name_of(self.source, node, spec)
            if name:
                kind = ("interface" if t in spec.interface_types else
                        "struct" if t in spec.struct_types else "type")
                self._emit_def(node, name, kind)
            # types have no call bodies worth walking for edges
            return

        if t in spec.function_types or t in spec.method_types:
            name = _name_of(self.source, node, spec)
            if name:
                is_method = self._class_depth > 0 and self._func_depth == 0
                # rust functions inside impl are methods; top-level are functions
                self._emit_def(node, name, "method" if is_method else "function")
                self._scope.append(name)
                self._func_depth += 1
                for c in node.children:
                    self._walk(c)
                self._func_depth -= 1
                self._scope.pop()
                return
            # unnamed callable (anonymous arrow/fn passed as arg) — walk for calls
            for c in node.children:
                self._walk(c)
            return

        for c in node.children:
            self._walk(c)

    def run(self, root) -> Optional[str]:
        """Walk the tree. Returns None on success, else a failure reason.

        NEVER RAISES — the module contract. A grammar whose node API disagrees
        with the walker (a tree-sitter core/pack version skew) used to throw an
        AttributeError straight out of `build_graph`, taking the WHOLE repo's
        graph — including the entirely-unaffected Python half — down with one
        bad file. Now the file is reported as partially extracted and everything
        else still indexes.
        """
        try:
            self._walk(root)
        except RecursionError:  # pragma: no cover - very deep trees
            logger.debug("ts_extract: recursion limit walking %s", self.relpath)
            return "walk hit the recursion limit"
        except Exception as exc:  # noqa: BLE001 - one bad file never fails the build
            logger.debug("ts_extract: walk failed for %s: %s", self.relpath, exc)
            return f"walk failed: {type(exc).__name__}: {exc}"
        return None


# ── Import table ─────────────────────────────────────────────────────────────


def _build_import_table(root, source: bytes, spec: LangSpec, relpath: str,
                        repo_files: Optional[set[str]]) -> dict[str, str]:
    """Map a local import alias → the relpath of the imported source file.

    Resolution is relative-path based (``./x`` / ``../y``) and only binds when
    the target file actually exists in the repo file set (``repo_files``). A
    bare/package import (``react``, ``lodash``) has no local target and is
    skipped — call_graph then resolves such calls by short-name over-report.
    """
    table: dict[str, str] = {}
    if repo_files is None:
        return table
    from pathlib import PurePosixPath
    here = PurePosixPath(relpath).parent

    def _resolve_module(mod: str) -> Optional[str]:
        if not mod.startswith("."):
            return None  # package import — no local file
        base = (here / mod)
        # normalise .. / .
        parts: list[str] = []
        for p in base.parts:
            if p == "..":
                if parts:
                    parts.pop()
            elif p not in (".", ""):
                parts.append(p)
        stem = "/".join(parts)
        for cand in (stem, f"{stem}.ts", f"{stem}.tsx", f"{stem}.js",
                     f"{stem}.jsx", f"{stem}.vue", f"{stem}/index.ts",
                     f"{stem}/index.js", f"{stem}.go", f"{stem}.rs"):
            if cand in repo_files:
                return cand
        return None

    def _walk(node):
        if node.type in spec.import_types:
            mod = None
            for c in node.children:
                if c.type == "string":
                    for g in c.children:
                        if g.type == "string_fragment":
                            mod = _node_text(source, g)
                elif c.type in ("interpreted_string_literal", "raw_string_literal"):
                    mod = _node_text(source, c).strip('"`')
            target = _resolve_module(mod) if mod else None
            if target:
                # Bind every imported identifier alias to the target file. When
                # the local alias RENAMES the declared symbol (`{ x as y }`),
                # encode the original name too (`"<file>::<orig>"`) — the def in
                # the target file is indexed under its OWN declared name, not
                # the alias, so a bare `y()` call site needs the original name
                # to find it (call_graph._resolve_callee step 2b).
                for local, orig in _import_aliases(node, source):
                    table[local] = target if (not orig or orig == local) else f"{target}::{orig}"
        for c in node.children:
            _walk(c)

    _walk(root)
    return table


def _import_aliases(import_node, source: bytes) -> list[tuple[str, str]]:
    """Local binding names introduced by an import statement, paired with the
    ORIGINAL declared name in the source module (named, default, namespace).

    The second element is "" when the alias IS the declared name (no rename)
    or the original name can't be determined (default/namespace imports —
    there is no separate declared symbol name to recover from the import
    site itself).
    """
    out: list[tuple[str, str]] = []

    def _walk(n):
        if n.type == "import_specifier":
            # `{ bar as baz }` — `name` is the ORIGINAL declared symbol,
            # `alias` (if present) is the local binding used at call sites.
            alias = n.child_by_field_name("alias")
            name = n.child_by_field_name("name")
            if name is None and alias is None:
                idents = [c for c in n.children if c.type == "identifier"]
                name = idents[0] if idents else None
                alias = idents[-1] if len(idents) > 1 else None
            chosen = alias or name
            if chosen is not None:
                local = _node_text(source, chosen)
                orig = _node_text(source, name) if name is not None else ""
                out.append((local, orig))
            return
        if n.type == "namespace_import":
            for c in n.children:
                if c.type == "identifier":
                    out.append((_node_text(source, c), ""))
            return
        for c in n.children:
            _walk(c)

    # default import: `import def from './y'` → import_clause has a bare identifier
    for c in import_node.children:
        if c.type == "import_clause":
            for g in c.children:
                if g.type == "identifier":
                    out.append((_node_text(source, g), ""))
    _walk(import_node)
    return out


# ── Public extraction entrypoints (return call_graph._FileExtract) ───────────


def _has_error(root) -> bool:
    """Whether the parse tree contains ERROR/MISSING nodes.

    `has_error` is a PROPERTY on modern tree-sitter bindings and a METHOD on
    older ones — reading it blindly would make `bool(<bound method>)` always
    True and mark every file parse_partial (a silent-wrong-answer, #158-class).
    """
    attr = getattr(root, "has_error", None)
    if attr is None:
        return False
    try:
        return bool(attr() if callable(attr) else attr)
    except Exception:  # noqa: BLE001 - a binding quirk must never fail the build
        return False


def _empty_extract(status: Optional[str] = None, reason: str = ""):
    """An empty extract, carrying a COVERAGE status so the caller can tell
    "nothing to index here" apart from "this file was never parsed"."""
    from .call_graph import _FileExtract, COVERAGE_INDEXED  # noqa: PLC0415
    return _FileExtract(defs=[], calls=[], import_table={},
                        status=status or COVERAGE_INDEXED, status_reason=reason)


def extract_ts_like(source: str, relpath: str, lang: str,
                    repo_files: Optional[set[str]] = None,
                    *, line_offset: int = 0, qual_prefix: str = ""):
    """Extract one non-Python source string into a call_graph._FileExtract.

    ``repo_files`` (the repo's relpath set) lets local relative imports resolve
    to a concrete file for high-confidence ``module.func`` edges; omit it and
    module-hinted calls fall back to short-name over-report (still correct,
    just lower confidence).
    """
    from .call_graph import (  # noqa: PLC0415
        _FileExtract, COVERAGE_INDEXED, COVERAGE_PARSE_PARTIAL, COVERAGE_SKIPPED,
    )
    if not is_tree_sitter_functional():
        return _empty_extract(COVERAGE_SKIPPED, "tree-sitter unavailable in this environment")
    spec = _SPEC_FOR_LANG.get(lang)
    if spec is None:
        return _empty_extract(COVERAGE_SKIPPED, f"no LangSpec for '{lang}'")
    try:
        parser = _tslp.get_parser(lang)
        data = source.encode("utf-8", errors="replace")
        tree = _parse(parser, data)
        root = tree.root_node
    except Exception as exc:  # noqa: BLE001 - never raise into the graph build
        logger.debug("ts_extract: %s parse failed for %s: %s", lang, relpath, exc)
        return _empty_extract(COVERAGE_SKIPPED, f"{lang} parse failed: {type(exc).__name__}")

    try:
        import_table = _build_import_table(root, data, spec, relpath, repo_files)
    except Exception as exc:  # noqa: BLE001 - a grammar skew must not fail the build
        logger.debug("ts_extract: import table failed for %s: %s", relpath, exc)
        import_table = {}
    visitor = _TSVisitor(relpath, data, spec, import_table,
                         line_offset=line_offset, qual_prefix=qual_prefix)
    walk_failure = visitor.run(root)
    # tree-sitter always returns a tree; an ERROR node in it means the extract
    # below is genuinely INCOMPLETE, not empty — say which (rule #158).
    if walk_failure:
        status, reason = COVERAGE_PARSE_PARTIAL, walk_failure
    elif _has_error(root):
        status, reason = COVERAGE_PARSE_PARTIAL, "tree-sitter reported syntax errors in this file"
    else:
        status, reason = COVERAGE_INDEXED, ""
    return _FileExtract(
        defs=visitor.defs, calls=visitor.calls, import_table=import_table,
        status=status, status_reason=reason,
    )


# ── Vue SFC ──────────────────────────────────────────────────────────────────
# Strategy ported from CodeGraph's VueExtractor (MIT): a .vue file yields a
# COMPONENT def for the file itself, its <script> block(s) are extracted and
# parsed as TS/JS (defs/calls attributed under "<Component>."), and template
# <Component> usages become call edges from the component to the child so a
# component used only in markup is still visible to callers/impact.

_SCRIPT_BLOCK_RE = re.compile(
    r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.DOTALL | re.IGNORECASE)
_COVERED_BLOCK_RE = re.compile(
    r"<(script|style)(\s[^>]*)?>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TEMPLATE_TAG_RE = re.compile(r"<([A-Za-z][A-Za-z0-9_.-]*)\b")

_VUE_BUILTINS = frozenset({
    "template", "component", "transition", "transitiongroup", "keepalive",
    "suspense", "teleport", "slot", "router-view", "router-link",
})
# Native HTML tags are not component references — a lowercase all-alpha tag with
# no hyphen and not a known component is treated as HTML and ignored. We keep
# PascalCase and kebab-case (hyphenated) tags as component refs.
_HTML_TAG_RE = re.compile(r"^[a-z][a-z0-9]*$")


def _stem(relpath: str) -> str:
    base = relpath.replace("\\", "/").split("/")[-1]
    return base[:-4] if base.lower().endswith(".vue") else base


def _kebab_to_pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[-_.]", name) if p)


def extract_vue(source: str, relpath: str,
                repo_files: Optional[set[str]] = None):
    """Extract a Vue SFC into a call_graph._FileExtract.

    Emits: a ``vue_component`` def for the file (always an entrypoint —
    components are imported/mounted externally); every <script> block's defs +
    calls (attributed under ``<Component>.`` so they don't collide with a same-
    named symbol elsewhere); and a call edge Component → each distinct child
    component used in the <template> (boundary-honest: a `{{ dynamic }}` or
    `:is` binding is NOT faked — only literal ``<Tag>`` usages are edged).
    """
    from .call_graph import (  # noqa: PLC0415
        _FileExtract, SymbolDef, COVERAGE_INDEXED, COVERAGE_PARSE_PARTIAL,
    )

    component = _kebab_to_pascal(_stem(relpath))
    defs: list = [SymbolDef(
        symbol=f"{relpath}::{component}", file=relpath, qualname=component,
        short_name=component, lineno=1, end_lineno=max(1, source.count("\n") + 1),
        kind="class",  # component == a class-like, instantiable node (entrypoint)
        is_entrypoint=True,
    )]
    calls: list = []
    import_table: dict[str, str] = {}

    if not is_tree_sitter_functional():
        # still emit the component node + template edges (regex, parser-free).
        # PARTIAL, not indexed: the <script> block's defs/calls are missing.
        _add_template_edges(source, relpath, component, calls)
        return _FileExtract(
            defs=defs, calls=calls, import_table={},
            status=COVERAGE_PARSE_PARTIAL,
            status_reason="tree-sitter unavailable — template edges only, no <script> symbols",
        )

    status, reason = COVERAGE_INDEXED, ""
    # ---- script block(s) → TS/JS extract, offset by the block's start line ----
    for m in _SCRIPT_BLOCK_RE.finditer(source):
        attrs = m.group("attrs") or ""
        body = m.group("body") or ""
        lang = "typescript" if re.search(r'lang\s*=\s*["\']ts', attrs, re.I) else "javascript"
        line_offset = source[:m.start("body")].count("\n")
        sub = extract_ts_like(
            body, relpath, lang, repo_files,
            line_offset=line_offset, qual_prefix=f"{component}.",
        )
        defs.extend(sub.defs)
        calls.extend(sub.calls)
        # merge the script's local imports so module.func edges resolve
        import_table.update(sub.import_table)
        # A bad <script> block makes the WHOLE component's extract partial —
        # the template edges survive but the script symbols may not.
        if sub.status != COVERAGE_INDEXED and status == COVERAGE_INDEXED:
            status, reason = COVERAGE_PARSE_PARTIAL, f"<script> block: {sub.status_reason}"

    _add_template_edges(source, relpath, component, calls)
    return _FileExtract(defs=defs, calls=calls, import_table=import_table,
                        status=status, status_reason=reason)


def _add_template_edges(source: str, relpath: str, component: str, calls: list) -> None:
    """Component → child-component call edges from literal <Tag> template usages."""
    from .call_graph import CallSite  # noqa: PLC0415
    covered: list[tuple[int, int]] = []
    for m in _COVERED_BLOCK_RE.finditer(source):
        start = source[:m.start()].count("\n")
        end = start + m.group(0).count("\n")
        covered.append((start, end))

    seen: set[str] = set()
    for lineno0, line in enumerate(source.split("\n")):
        if any(s <= lineno0 <= e for s, e in covered):
            continue
        for tm in _TEMPLATE_TAG_RE.finditer(line):
            tag = tm.group(1)
            low = tag.lower()
            if low in _VUE_BUILTINS:
                continue
            # native HTML (lowercase, no hyphen, not a component) → ignore
            if _HTML_TAG_RE.match(tag) and "-" not in tag:
                continue
            child = _kebab_to_pascal(tag)
            if not child or child == component or child in seen:
                continue
            seen.add(child)
            calls.append(CallSite(
                caller=f"{relpath}::{component}",
                callee_short=child,
                n_positional=0, has_star=False, keywords=(),
                hint_self=False, hint_module=None,
                lineno=lineno0 + 1,
            ))
