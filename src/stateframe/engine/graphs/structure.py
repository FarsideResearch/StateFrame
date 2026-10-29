"""graphs/structure.py — the STRUCTURE layer (CanonicalLayer.STRUCTURE).

Answers a_graph_layers.md L1's questions: what files/classes/functions exist,
what imports what, which functions call which, where is a function defined.

Full build, both repos. Cost class C0/C1: one ``ast.parse`` per file plus one
walk. No import execution, no runtime introspection — closed-world static
analysis exactly as w65/a_wave2_prep pins it ("No importing or executing
FastAPI, Vue, workers, or migrations").

RESOLUTION HONESTY (a_graph_layers line ~172, "Dynamic or unresolved calls must
be represented explicitly. Dropping them creates false certainty."):

  * a call to a module-local ``def``/``class`` -> ``resolved``
  * a call to a name bound by an ``import``/``from ... import`` in the same file,
    whose target module is in the corpus and defines that symbol -> ``resolved``
  * anything else (bare unknown name, attribute chain, dynamic dispatch) is
    emitted anyway against an ``external:<name>`` placeholder entity with
    ``resolution=UNRESOLVED``. Never dropped, never guessed.

The unresolved count rides in the receipt.
"""

from __future__ import annotations

import ast
import time
from typing import Dict, List, Optional, Set, Tuple

from graphs.corpus import Corpus
from graphs.ids import EntityIdMinter
from graphs.model import (
    CONF_CERTAIN,
    CONF_LIKELY,
    CanonicalLayer,
    Edge,
    Evidence,
    Freshness,
    KnowledgeCut,
    LayerGraph,
    Node,
    RESOLVED,
    UNRESOLVED,
)

LAYER = CanonicalLayer.STRUCTURE.value

# Relation vocabulary — the subset of a_graph_layers L1 edges provable statically.
CONTAINS = "contains"
DECLARES = "declares"
IMPORTS = "imports"
CALLS = "calls"
INHERITS = "inherits"

FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

#: An unresolved call target. Deliberately NOT a canonical entity id — it is a
#: sentinel a consumer must recognise as "we could not bind this", which is
#: exactly why it does not parse as a domain id.
EXTERNAL_PREFIX = "external:"


def module_dotted(rel_path: str) -> str:
    """``api/support.py`` -> ``api.support``; ``api/__init__.py`` -> ``api``."""
    stem = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def dotted_candidates(dotted: str) -> Tuple[str, str]:
    """Candidate rel paths for a dotted module: ``a/b.py`` and ``a/b/__init__.py``."""
    p = dotted.replace(".", "/")
    return (p + ".py", p + "/__init__.py")


def qualnames(tree: ast.AST) -> List[Tuple[str, ast.AST, Optional[str]]]:
    """Every function/class in ``tree`` as ``(qualname, node, parent_qualname)``.

    Qualnames are dotted (``Klass.method``) so a method is distinguishable from a
    module-level function of the same name — the exact trap that would make two
    entities collide onto one ID.
    """
    out: List[Tuple[str, ast.AST, Optional[str]]] = []

    def walk(node: ast.AST, prefix: str, parent: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*FUNC_NODES, ast.ClassDef)):
                qn = f"{prefix}{child.name}"
                out.append((qn, child, parent))
                walk(child, qn + ".", qn)
            else:
                walk(child, prefix, parent)

    walk(tree, "", None)
    return out


class _FileSymbols:
    """Per-file symbol table: local defs plus import bindings. Static only."""

    __slots__ = ("local_defs", "import_bindings", "star_imports")

    def __init__(self) -> None:
        #: bare name -> qualname defined in this file
        self.local_defs: Dict[str, str] = {}
        #: bound name -> (dotted_module, original_symbol_or_None)
        self.import_bindings: Dict[str, Tuple[str, Optional[str]]] = {}
        #: modules star-imported; every name they could bind is UNPROVABLE, so a
        #: call resolving through one stays UNRESOLVED rather than being guessed.
        self.star_imports: Set[str] = set()


def _collect_symbols(tree: ast.AST, rel_path: str) -> _FileSymbols:
    syms = _FileSymbols()
    for qn, node, _parent in qualnames(tree):
        if "." not in qn:  # only module-level names are callable bare
            syms.local_defs[qn] = qn
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                syms.import_bindings[bound] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            dotted = _absolutize(node, rel_path)
            if dotted is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    syms.star_imports.add(dotted)
                    continue
                bound = alias.asname or alias.name
                syms.import_bindings[bound] = (dotted, alias.name)
    return syms


def _absolutize(node: ast.ImportFrom, rel_path: str) -> Optional[str]:
    """Resolve a possibly-relative ``from . import x`` against the file's package.

    Returns None when the relative level walks above the subtree root — that is
    an import we genuinely cannot bind inside this corpus, and the caller keeps
    it unresolved rather than inventing a module.
    """
    if not node.level:
        return node.module or None
    pkg_parts = rel_path.split("/")[:-1]
    up = node.level - 1
    if up > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - up] if up else pkg_parts
    tail = (node.module or "").split(".") if node.module else []
    parts = [p for p in [*base, *tail] if p]
    return ".".join(parts) if parts else None


def _called_name(call: ast.Call) -> Optional[str]:
    """The bare name a call targets, or None for a chain we will not guess.

    ``foo()`` -> ``foo``; ``mod.foo()`` -> ``mod.foo``; ``a.b.c()`` -> ``a.b.c``;
    ``obj().m()`` / ``d["k"]()`` -> None (dynamic, stays unresolved).
    """
    func = call.func
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None


def build_structure(
    corpus: Corpus,
    minter: EntityIdMinter,
    cut: KnowledgeCut,
) -> LayerGraph:
    """Full STRUCTURE build over one corpus. Deterministic, sorted throughout."""
    started = time.monotonic()
    graph = LayerGraph(layer=LAYER, cut=cut)
    receipt = graph.receipt
    receipt.files_seen = corpus.files_seen
    if corpus.truncated:
        receipt.truncated = True
        receipt.notes = receipt.notes + ("corpus file cap applied",)
    if corpus.read_failed:
        receipt.notes = receipt.notes + (f"unreadable files: {corpus.read_failed}",)

    trees: Dict[str, ast.AST] = {}
    symbols: Dict[str, _FileSymbols] = {}
    #: dotted module -> rel_path, for import resolution
    module_index: Dict[str, str] = {}
    #: rel_path -> {bare name: qualname} of module-level defs
    defs_by_file: Dict[str, Dict[str, str]] = {}

    # ---- pass 1: parse and index -------------------------------------
    for rel, src in corpus.items():
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            receipt.files_parse_failed += 1
            continue
        receipt.files_parsed += 1
        trees[rel] = tree
        module_index[module_dotted(rel)] = rel
        syms = _collect_symbols(tree, rel)
        symbols[rel] = syms
        defs_by_file[rel] = dict(syms.local_defs)

    # ---- pass 2: nodes -----------------------------------------------
    entity_of: Dict[Tuple[str, str], str] = {}  # (rel, qualname) -> entity id
    file_entity: Dict[str, str] = {}

    for rel in sorted(trees):
        fid = minter.file_id(rel)
        file_entity[rel] = fid
        graph.nodes.append(
            Node(
                layer=LAYER,
                entity_id=fid,
                kind="file",
                evidence=Evidence(rel, 1),
                attrs={"module": module_dotted(rel)},
            )
        )
        for qn, node, parent in qualnames(trees[rel]):
            is_class = isinstance(node, ast.ClassDef)
            eid = (minter.class_id if is_class else minter.function_id)(rel, qn)
            entity_of[(rel, qn)] = eid
            graph.nodes.append(
                Node(
                    layer=LAYER,
                    entity_id=eid,
                    kind="class" if is_class else "function",
                    evidence=Evidence(rel, getattr(node, "lineno", 0)),
                    attrs={
                        "qualname": qn,
                        "start_line": getattr(node, "lineno", 0),
                        "end_line": getattr(node, "end_lineno", 0) or 0,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    },
                )
            )
            owner = file_entity[rel] if parent is None else entity_of[(rel, parent)]
            graph.edges.append(
                Edge(
                    layer=LAYER,
                    src=owner,
                    relation=CONTAINS if parent else DECLARES,
                    dst=eid,
                    evidence=Evidence(rel, getattr(node, "lineno", 0)),
                )
            )

    # ---- pass 3: import edges ----------------------------------------
    for rel in sorted(trees):
        syms = symbols[rel]
        src_id = file_entity[rel]
        seen: Set[Tuple[str, str]] = set()
        for bound in sorted(syms.import_bindings):
            dotted, _sym = syms.import_bindings[bound]
            target_rel = _resolve_module(dotted, module_index)
            key = (dotted, target_rel or "")
            if key in seen:
                continue
            seen.add(key)
            if target_rel is not None:
                graph.edges.append(
                    Edge(
                        layer=LAYER,
                        src=src_id,
                        relation=IMPORTS,
                        dst=file_entity[target_rel],
                        evidence=Evidence(rel, 0),
                        attrs={"module": dotted},
                    )
                )
            else:
                graph.edges.append(
                    Edge(
                        layer=LAYER,
                        src=src_id,
                        relation=IMPORTS,
                        dst=EXTERNAL_PREFIX + dotted,
                        resolution=UNRESOLVED,
                        confidence=CONF_LIKELY,
                        evidence=Evidence(rel, 0),
                        attrs={"module": dotted, "reason": "module outside corpus"},
                    )
                )

    # ---- pass 4: call edges ------------------------------------------
    for rel in sorted(trees):
        syms = symbols[rel]
        local = defs_by_file[rel]
        for qn, node, _parent in qualnames(trees[rel]):
            if not isinstance(node, FUNC_NODES):
                continue
            caller = entity_of[(rel, qn)]
            emitted: Set[Tuple[str, str]] = set()
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                name = _called_name(sub)
                line = getattr(sub, "lineno", 0)
                if name is None:
                    _emit_unresolved(
                        graph, caller, "<dynamic>", rel, line, "non-name callee"
                    )
                    continue
                target = _resolve_call(name, rel, local, syms, module_index, defs_by_file)
                if target is None:
                    if (name, "u") in emitted:
                        continue
                    emitted.add((name, "u"))
                    reason = (
                        "star-import shadow"
                        if syms.star_imports
                        else "unbound name"
                    )
                    _emit_unresolved(graph, caller, name, rel, line, reason)
                    continue
                t_rel, t_qn = target
                dst = entity_of.get((t_rel, t_qn))
                if dst is None:
                    if (name, "u") in emitted:
                        continue
                    emitted.add((name, "u"))
                    _emit_unresolved(graph, caller, name, rel, line, "target not indexed")
                    continue
                if (name, dst) in emitted:
                    continue
                emitted.add((name, dst))
                graph.edges.append(
                    Edge(
                        layer=LAYER,
                        src=caller,
                        relation=CALLS,
                        dst=dst,
                        resolution=RESOLVED,
                        confidence=CONF_CERTAIN,
                        evidence=Evidence(rel, line),
                        attrs={"name": name},
                    )
                )

    # ---- pass 5: inheritance -----------------------------------------
    for rel in sorted(trees):
        syms = symbols[rel]
        local = defs_by_file[rel]
        for qn, node, _parent in qualnames(trees[rel]):
            if not isinstance(node, ast.ClassDef):
                continue
            child = entity_of[(rel, qn)]
            for base in node.bases:
                name = _called_name(ast.Call(func=base, args=[], keywords=[]))
                if name is None:
                    continue
                target = _resolve_call(name, rel, local, syms, module_index, defs_by_file)
                if target is None or (target[0], target[1]) not in entity_of:
                    graph.edges.append(
                        Edge(
                            layer=LAYER,
                            src=child,
                            relation=INHERITS,
                            dst=EXTERNAL_PREFIX + name,
                            resolution=UNRESOLVED,
                            confidence=CONF_LIKELY,
                            evidence=Evidence(rel, getattr(node, "lineno", 0)),
                        )
                    )
                else:
                    graph.edges.append(
                        Edge(
                            layer=LAYER,
                            src=child,
                            relation=INHERITS,
                            dst=entity_of[(target[0], target[1])],
                            evidence=Evidence(rel, getattr(node, "lineno", 0)),
                        )
                    )

    receipt.id_collisions = len(minter.collisions)
    receipt.build_seconds = round(time.monotonic() - started, 3)
    graph.freshness = Freshness.CURRENT
    return graph.finalize()


def _emit_unresolved(
    graph: LayerGraph, caller: str, name: str, rel: str, line: int, reason: str
) -> None:
    graph.edges.append(
        Edge(
            layer=LAYER,
            src=caller,
            relation=CALLS,
            dst=EXTERNAL_PREFIX + name,
            resolution=UNRESOLVED,
            confidence=CONF_LIKELY,
            evidence=Evidence(rel, line),
            attrs={"name": name, "reason": reason},
        )
    )


def _resolve_module(dotted: str, module_index: Dict[str, str]) -> Optional[str]:
    """Dotted module -> rel path in this corpus, or None (outside the closed world).

    Tries the full dotted name first, then progressively strips a leading
    package segment: a corpus rooted at ``backend/app`` sees ``app.api.support``
    as ``api/support.py``.
    """
    if dotted in module_index:
        return module_index[dotted]
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        cand = ".".join(parts[i:])
        if cand in module_index:
            return module_index[cand]
    return None


def _resolve_call(
    name: str,
    rel: str,
    local: Dict[str, str],
    syms: _FileSymbols,
    module_index: Dict[str, str],
    defs_by_file: Dict[str, Dict[str, str]],
) -> Optional[Tuple[str, str]]:
    """Bind a called name to ``(rel_path, qualname)``, or None if unprovable."""
    head, _, tail = name.partition(".")

    # 1. module-local definition, called bare
    if not tail and head in local:
        return (rel, local[head])

    binding = syms.import_bindings.get(head)
    if binding is None:
        return None
    dotted, orig = binding

    # 2. `from mod import fn` -> fn()
    if orig is not None and not tail:
        target_rel = _resolve_module(dotted, module_index)
        if target_rel is None:
            return None
        if orig in defs_by_file.get(target_rel, {}):
            return (target_rel, orig)
        return None

    # 3. `import mod` -> mod.fn()  /  `from pkg import mod` -> mod.fn()
    if tail and "." not in tail:
        mod_dotted = dotted if orig is None else f"{dotted}.{orig}"
        target_rel = _resolve_module(mod_dotted, module_index)
        if target_rel is None:
            return None
        if tail in defs_by_file.get(target_rel, {}):
            return (target_rel, tail)
        return None

    return None


__all__ = [
    "CALLS",
    "CONTAINS",
    "DECLARES",
    "EXTERNAL_PREFIX",
    "IMPORTS",
    "INHERITS",
    "LAYER",
    "build_structure",
    "dotted_candidates",
    "module_dotted",
    "qualnames",
]
