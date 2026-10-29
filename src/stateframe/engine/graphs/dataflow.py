"""graphs/dataflow.py — the DATAFLOW layer, LAZY per-module.

w65/a_wave2_prep names this lane's specific risk:

    "The L2 dataflow risk is larger than runtime. If the 3,300-file workload
     forces shortcuts, the graph may become cheaper but semantically
     inconsistent. ... L2 dataflow should reuse the existing bounded indexes and
     memoization rather than launch an independent whole-repository traversal."

So DATAFLOW is **never built eagerly**. ``DataflowLayer`` is a materializer:
nothing is computed until a caller asks for a specific module, and what it
computes is memoized per module for the life of the object. A whole-repo
dataflow pass is not merely discouraged here — there is no function that does
one. ``materialize_all`` does not exist; ``materialized_modules`` reports how
much of the repo a session actually touched, and the receipt records it.

Scope, stated rather than implied: this is INTRAPROCEDURAL def-use only.
Definitions are the binding constructs Python itself defines (assignment
targets, ``for`` targets, ``with ... as``, ``except ... as``, comprehension
targets, walrus, parameters, nested def/class names, imports). Uses are
``ast.Name`` loads and the attribute roots of loads. Interprocedural summaries,
alias analysis, and taint are OUT of v1 and listed in ``model.KNOWN_GAPS`` —
a caller must not mistake the absence of a taint edge for the absence of taint.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from graphs.corpus import Corpus
from graphs.ids import EntityIdMinter
from graphs.model import (
    CONF_CERTAIN,
    CanonicalLayer,
    Edge,
    Evidence,
    Freshness,
    KnowledgeCut,
    LayerGraph,
    Node,
    PARTIAL,
    UNRESOLVED,
)
from graphs.structure import FUNC_NODES, qualnames

LAYER = CanonicalLayer.DATAFLOW.value

DEFINES = "defines"
USES = "uses"
FLOWS_TO = "flows_to"

#: Statements per function past which we stop enumerating flows and say so.
#: A cap is REPORTED, never silently applied (edit_sets.ProviderIndex idiom).
MAX_STATEMENTS_PER_FUNCTION = 400


@dataclass(frozen=True)
class Definition:
    name: str
    line: int
    construct: str


@dataclass(frozen=True)
class Use:
    name: str
    line: int


@dataclass(frozen=True)
class FunctionDataflow:
    """Def-use for ONE function. The unit of lazy materialization."""

    entity_id: str
    rel_path: str
    qualname: str
    definitions: Tuple[Definition, ...]
    uses: Tuple[Use, ...]
    #: (def_line, use_line, name) — a use reached by the nearest preceding def
    #: of the same name in the same function. Line-ordered, not path-sensitive:
    #: this is a def-use approximation and is labelled ``PARTIAL`` on the edge.
    reaching: Tuple[Tuple[int, int, str], ...]
    truncated: bool = False

    @property
    def free_names(self) -> Tuple[str, ...]:
        """Names used but never defined in this function — the interprocedural
        boundary. v1 does not chase them; it names them."""
        defined = {d.name for d in self.definitions}
        return tuple(sorted({u.name for u in self.uses} - defined))


def _definitions(fn: ast.AST) -> List[Definition]:
    out: List[Definition] = []
    args = getattr(fn, "args", None)
    if args is not None:
        for a in (
            list(getattr(args, "posonlyargs", []))
            + list(args.args)
            + list(args.kwonlyargs)
        ):
            out.append(Definition(a.arg, getattr(a, "lineno", 0), "parameter"))
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                out.append(Definition(extra.arg, getattr(extra, "lineno", 0), "parameter"))

    for node in ast.walk(fn):
        if node is fn:
            continue
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            out.append(Definition(node.id, line, "assign"))
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            out.append(Definition(node.target.id, line, "walrus"))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.append(Definition(node.name, line, "except_as"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or (
                    alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
                )
                out.append(Definition(bound, line, "import"))
        elif isinstance(node, (*FUNC_NODES, ast.ClassDef)):
            out.append(Definition(node.name, line, "nested_def"))
        elif isinstance(node, ast.Global):
            for n in node.names:
                out.append(Definition(n, line, "global"))
        elif isinstance(node, ast.Nonlocal):
            for n in node.names:
                out.append(Definition(n, line, "nonlocal"))
    return out


def _uses(fn: ast.AST) -> List[Use]:
    out: List[Use] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.append(Use(node.id, getattr(node, "lineno", 0)))
    return out


def _reaching(defs: List[Definition], uses: List[Use]) -> List[Tuple[int, int, str]]:
    """Nearest preceding definition per use, by line. Deterministic.

    This is an APPROXIMATION of reaching definitions (no CFG, no branch
    merging). Every edge derived from it is emitted with ``resolution=PARTIAL``
    so a consumer can never read it as a proven flow.
    """
    by_name: Dict[str, List[int]] = {}
    for d in sorted(defs, key=lambda d: (d.name, d.line)):
        by_name.setdefault(d.name, []).append(d.line)
    out: List[Tuple[int, int, str]] = []
    for u in sorted(uses, key=lambda u: (u.line, u.name)):
        lines = by_name.get(u.name)
        if not lines:
            continue
        prior = [ln for ln in lines if ln <= u.line]
        # a parameter (lineno == def line) reaches uses on the same line too
        chosen = prior[-1] if prior else lines[0]
        out.append((chosen, u.line, u.name))
    return sorted(set(out))


def analyze_function(fn: ast.AST, entity_id: str, rel_path: str, qualname: str) -> FunctionDataflow:
    """Def-use for one function. Pure; no I/O; deterministic."""
    stmt_count = sum(1 for n in ast.walk(fn) if isinstance(n, ast.stmt))
    truncated = stmt_count > MAX_STATEMENTS_PER_FUNCTION
    defs = _definitions(fn)
    uses = _uses(fn)
    reaching = () if truncated else tuple(_reaching(defs, uses))
    return FunctionDataflow(
        entity_id=entity_id,
        rel_path=rel_path,
        qualname=qualname,
        definitions=tuple(sorted(defs, key=lambda d: (d.line, d.name, d.construct))),
        uses=tuple(sorted(set(uses), key=lambda u: (u.line, u.name))),
        reaching=reaching,
        truncated=truncated,
    )


class DataflowLayer:
    """Lazy per-module dataflow materializer.

    There is deliberately NO whole-repo build entrypoint. A caller materializes
    the modules it needs; ``receipt()`` reports exactly how many that was, so a
    session that touched 4 of 4,453 modules cannot be reported as a repo-wide
    dataflow graph.
    """

    def __init__(self, corpus: Corpus, minter: EntityIdMinter, cut: KnowledgeCut):
        self._corpus = corpus
        self._minter = minter
        self._cut = cut
        self._cache: Dict[str, Dict[str, FunctionDataflow]] = {}
        self._parse_failed: Set[str] = set()
        self._seconds = 0.0

    # -- the lazy surface ---------------------------------------------

    def materialize_module(self, rel_path: str) -> Dict[str, FunctionDataflow]:
        """Def-use for every function in one module. Memoized.

        Raises ``KeyError`` for a path outside the corpus — a silent empty dict
        would look identical to "this module has no functions".
        """
        if rel_path in self._cache:
            return self._cache[rel_path]
        if rel_path not in self._corpus.files:
            raise KeyError(f"{rel_path!r} is not in corpus {self._corpus.repo_key!r}")
        started = time.monotonic()
        try:
            tree = ast.parse(self._corpus.files[rel_path])
        except (SyntaxError, ValueError, RecursionError):
            self._parse_failed.add(rel_path)
            self._cache[rel_path] = {}
            return self._cache[rel_path]
        out: Dict[str, FunctionDataflow] = {}
        for qn, node, _parent in qualnames(tree):
            if not isinstance(node, FUNC_NODES):
                continue
            eid = self._minter.function_id(rel_path, qn)
            out[qn] = analyze_function(node, eid, rel_path, qn)
        self._cache[rel_path] = out
        self._seconds += time.monotonic() - started
        return out

    def for_function(self, rel_path: str, qualname: str) -> Optional[FunctionDataflow]:
        return self.materialize_module(rel_path).get(qualname)

    @property
    def materialized_modules(self) -> Tuple[str, ...]:
        return tuple(sorted(self._cache))

    @property
    def is_eager(self) -> bool:
        """Always False. Pinned by a test so no future edit can quietly make the
        layer eager and reintroduce the wave2_prep risk."""
        return False

    # -- projection into the shared fact format ------------------------

    def to_layer_graph(self) -> LayerGraph:
        """Project ONLY what has been materialized into nodes/edges.

        The receipt's ``sampled`` flag is set whenever fewer modules were
        materialized than the corpus holds — which is the normal case, and must
        never read as a complete dataflow graph.
        """
        graph = LayerGraph(layer=LAYER, cut=self._cut)
        for rel in sorted(self._cache):
            for qn in sorted(self._cache[rel]):
                fdf = self._cache[rel][qn]
                graph.nodes.append(
                    Node(
                        layer=LAYER,
                        entity_id=fdf.entity_id,
                        kind="function_dataflow",
                        evidence=Evidence(rel, fdf.definitions[0].line if fdf.definitions else 0),
                        attrs={
                            "qualname": qn,
                            "definitions": len(fdf.definitions),
                            "uses": len(fdf.uses),
                            "free_names": list(fdf.free_names),
                            "truncated": fdf.truncated,
                        },
                    )
                )
                for def_line, use_line, name in fdf.reaching:
                    graph.edges.append(
                        Edge(
                            layer=LAYER,
                            src=fdf.entity_id,
                            relation=FLOWS_TO,
                            dst=fdf.entity_id,
                            resolution=PARTIAL,
                            confidence=CONF_CERTAIN,
                            evidence=Evidence(rel, use_line),
                            attrs={"name": name, "def_line": def_line, "use_line": use_line},
                        )
                    )
                for free in fdf.free_names:
                    graph.edges.append(
                        Edge(
                            layer=LAYER,
                            src=fdf.entity_id,
                            relation=USES,
                            dst="external:" + free,
                            resolution=UNRESOLVED,
                            evidence=Evidence(rel, 0),
                            attrs={"name": free, "reason": "free name (interprocedural)"},
                        )
                    )
        r = graph.receipt
        r.files_seen = self._corpus.files_seen
        r.files_parsed = len(self._cache) - len(self._parse_failed)
        r.files_parse_failed = len(self._parse_failed)
        r.build_seconds = round(self._seconds, 3)
        r.sampled = len(self._cache) < len(self._corpus.files)
        if r.sampled:
            r.sample_note = (
                f"lazy: {len(self._cache)} of {len(self._corpus.files)} modules materialized"
            )
        if any(f.truncated for m in self._cache.values() for f in m.values()):
            r.truncated = True
            r.notes = r.notes + (
                f"functions over {MAX_STATEMENTS_PER_FUNCTION} statements skipped reaching-def",
            )
        graph.freshness = Freshness.CURRENT if self._cache else Freshness.UNAVAILABLE
        return graph.finalize()


__all__ = [
    "DEFINES",
    "DataflowLayer",
    "Definition",
    "FLOWS_TO",
    "FunctionDataflow",
    "LAYER",
    "MAX_STATEMENTS_PER_FUNCTION",
    "USES",
    "Use",
    "analyze_function",
]
