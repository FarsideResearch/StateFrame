"""Relational substrate v1 — one typed relation store, one query contract.

Committed design (research answer `/tmp/solq/a_relational.md`):

  * ONE logical relation ledger with a GOVERNED vocabulary and ONE query
    contract; the silos stay native and publish into it through adapters.
    Not a monolithic canonical KG (schema explosion), not N independently
    queryable graphs (join blindness).
  * Every atom carries provenance (which engine asserted it, by what method)
    and confidence: 1.0 for statically derived facts, < 1.0 for asserted ones.
  * Every query returns evidence — the relations traversed, the confidence of
    the derivation, the depth — never bare nodes.

The first two silos unified here are exactly the two the research answer
names as the v1 proof: the CODE GRAPH (``func_index.build_func_index``) and
the SEAM MANIFEST (``chains.seam_manifest``). They share stable path/qualname
identifiers, so ``function -> file -> definition -> reference site`` becomes a
single closure query instead of two hand-joined dictionaries.

Pure stdlib. Every query verb returns a deterministically ordered result: the
store is a decision input for a deterministic kernel, so two stores holding
the same atoms MUST answer identically regardless of insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

# --------------------------------------------------------------- vocabulary

#: The governed core vocabulary. A relation type outside this frozenset is
#: REJECTED at add() time — an ungoverned type has no composition rule, so it
#: would silently fall out of every closure and reasoning primitive rather
#: than failing loudly (the silent-drop class).
RELATION_TYPES = frozenset({
    "contains",
    "calls",
    "defines",
    "imports",
    "consumes",
    "precedes",
    "causes",
    "resembles",
    "depends_on",
    "part_of",
    "references",
})

#: Inverse pairs, for direction-aware traversal and future consistency checks.
INVERSES = {
    "contains": "part_of",
    "part_of": "contains",
}

#: The TYPE ALGEBRA. ``(left, right) -> composed`` — a path whose consecutive
#: edge types compose to a single type is a derivable relation of that type.
#: Anything absent from this table does NOT compose, which is the point:
#: ``resembles`` is deliberately absent, so a chain of resemblances never
#: becomes a resemblance. (X resembles Y, Y resembles Z does NOT make X
#: resemble Z — the classic transitivity fallacy for similarity relations.)
COMPOSITION = {
    ("contains", "contains"): "contains",
    ("part_of", "part_of"): "part_of",
    ("causes", "causes"): "causes",
    ("precedes", "precedes"): "precedes",
    ("depends_on", "depends_on"): "depends_on",
    ("imports", "imports"): "imports",
}

#: Types whose composition DECAYS confidence multiplicatively: each additional
#: causal hop is strictly weaker evidence than the hop before it. Structural
#: containment does not decay — a file containing a class containing a method
#: is as certain at depth 3 as at depth 1 — so those take the conservative
#: min() of the path instead.
DECAYING = frozenset({"causes"})

#: Types that are never transitive, asserted explicitly so a future edit that
#: adds them to COMPOSITION has to delete them here first.
NON_TRANSITIVE = frozenset({"resembles", "calls", "defines", "references",
                            "consumes"})

#: Default traversal set for impact(): "if x changed/broke, who feels it?"
#: These are the dependency-bearing edges, followed in REVERSE.
IMPACT_TYPES = frozenset({"depends_on", "calls", "imports", "consumes",
                          "references", "contains"})

STATIC_CONFIDENCE = 1.0


class RelationTypeError(ValueError):
    """Raised when a relation type is outside the governed vocabulary."""


def compose(left: str, right: str) -> Optional[str]:
    """The composed type of ``left`` followed by ``right``, or None.

    None means the two edges do not chain into a single derivable relation —
    the traversal stops rather than inventing an untyped implication.
    """
    return COMPOSITION.get((left, right))


def compose_confidence(relation_type: str, left: float, right: float) -> float:
    """Confidence of a composed edge. Decaying types multiply, others min()."""
    if relation_type in DECAYING:
        return left * right
    return min(left, right)


# ------------------------------------------------------------------ records

@dataclass(frozen=True)
class Relation:
    """One relation atom.

    source/target are opaque stable IDs (this substrate never parses them;
    the adapters own ID minting). provenance names the asserting engine AND
    its method, e.g. ``"seam_manifest:reference:call"`` — "which engine said
    so, how" is the whole point of keeping a ledger rather than a graph.
    """

    source: str
    relation_type: str
    target: str
    provenance: str
    confidence: float = STATIC_CONFIDENCE
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    def __post_init__(self):
        if self.relation_type not in RELATION_TYPES:
            raise RelationTypeError(
                f"{self.relation_type!r} is not in the governed vocabulary; "
                f"known types: {sorted(RELATION_TYPES)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence {self.confidence!r} outside [0.0, 1.0]")
        if not self.source or not self.target:
            raise ValueError("source and target must be non-empty ids")
        if not self.provenance:
            raise ValueError("every relation must carry provenance")

    @property
    def key(self) -> tuple:
        """Identity of the ASSERTION, ignoring how strongly it is believed."""
        return (self.source, self.relation_type, self.target,
                self.valid_from, self.valid_to)

    @property
    def sort_key(self) -> tuple:
        return (self.source, self.relation_type, self.target, self.provenance)

    def as_dict(self) -> dict:
        return dict(source=self.source, relation_type=self.relation_type,
                    target=self.target, provenance=self.provenance,
                    confidence=self.confidence, valid_from=self.valid_from,
                    valid_to=self.valid_to)


@dataclass(frozen=True)
class Derivation:
    """A closure/impact hit: the node reached, plus the proof it was reached.

    ``relation_type`` is the COMPOSED type (None for reachability-mode walks
    such as impact()); ``path`` is the exact chain of atoms traversed, so
    every answer this substrate gives can be replayed by hand.
    """

    target: str
    relation_type: Optional[str]
    confidence: float
    depth: int
    path: tuple  # tuple[Relation, ...]

    @property
    def sort_key(self) -> tuple:
        return (self.depth, self.target, self.relation_type or "")

    def as_dict(self) -> dict:
        return dict(target=self.target, relation_type=self.relation_type,
                    confidence=self.confidence, depth=self.depth,
                    path=[r.as_dict() for r in self.path])


# -------------------------------------------------------------------- store

@dataclass
class RelationStore:
    """The relation ledger plus the query contract every engine gets."""

    _by_key: dict = field(default_factory=dict)
    _out: dict = field(default_factory=dict)   # source -> [Relation]
    _in: dict = field(default_factory=dict)    # target -> [Relation]

    # -- ingest ------------------------------------------------------------
    def add(self, relation: Relation) -> Relation:
        """Add one atom. A duplicate assertion MERGES, keeping max confidence.

        Two engines deriving the same static fact is corroboration, not
        conflict: we keep the strongest belief and record the merged
        provenance so the ledger still names every asserter. Provenance is
        merged in sorted order so the merge is order-independent.
        """
        if not isinstance(relation, Relation):
            raise TypeError("add() takes a Relation")
        existing = self._by_key.get(relation.key)
        if existing is None:
            self._by_key[relation.key] = relation
            self._out.setdefault(relation.source, []).append(relation)
            self._in.setdefault(relation.target, []).append(relation)
            return relation

        provs = sorted(set(existing.provenance.split("|"))
                       | set(relation.provenance.split("|")))
        merged = replace(
            existing,
            confidence=max(existing.confidence, relation.confidence),
            provenance="|".join(provs))
        self._replace_atom(existing, merged)
        return merged

    def add_all(self, relations: Iterable[Relation]) -> None:
        for r in relations:
            self.add(r)

    def _replace_atom(self, old: Relation, new: Relation) -> None:
        self._by_key[new.key] = new
        for bucket, node in ((self._out, old.source), (self._in, old.target)):
            lst = bucket[node]
            lst[lst.index(old)] = new

    # -- inspection --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_key)

    def relations(self) -> list:
        """Every atom, deterministically ordered."""
        return sorted(self._by_key.values(), key=lambda r: r.sort_key)

    def nodes(self) -> list:
        out = set(self._out) | set(self._in)
        return sorted(out)

    def fingerprint(self) -> tuple:
        """Order-independent identity of the ledger's contents."""
        return tuple(r.as_dict()["source"] + "\x1f" + r.relation_type
                     + "\x1f" + r.target + "\x1f" + r.provenance
                     + "\x1f" + f"{r.confidence:.6f}"
                     for r in self.relations())

    # -- query verb: neighbors --------------------------------------------
    def neighbors(self, x: str, types=None, direction: str = "out") -> list:
        """Atoms incident to ``x``. direction: 'out' | 'in' | 'both'."""
        if direction not in ("out", "in", "both"):
            raise ValueError(f"unknown direction {direction!r}")
        allowed = self._normalize_types(types)
        found = []
        if direction in ("out", "both"):
            found.extend(self._out.get(x, ()))
        if direction in ("in", "both"):
            found.extend(self._in.get(x, ()))
        seen, out = set(), []
        for r in found:
            if allowed is not None and r.relation_type not in allowed:
                continue
            if r.key in seen:
                continue
            seen.add(r.key)
            out.append(r)
        return sorted(out, key=lambda r: r.sort_key)

    # -- query verb: closure ----------------------------------------------
    def closure(self, x: str, types=None, depth: int = 3) -> list:
        """Type-composed transitive closure from ``x``.

        Only chains whose consecutive edge types COMPOSE are followed, so
        ``resembles`` never chains and ``contains ∘ contains`` does. Confidence
        follows the type algebra (multiplicative for causes, min elsewhere).
        The best (highest-confidence, then shallowest) derivation per reached
        node wins.
        """
        allowed = self._normalize_types(types)
        if depth < 1:
            return []
        best: dict = {}
        # frontier entries: (node, composed_type, confidence, path)
        frontier = [(x, None, STATIC_CONFIDENCE, ())]
        for level in range(1, depth + 1):
            nxt = []
            for node, ctype, conf, path in frontier:
                for r in self.neighbors(node, allowed, direction="out"):
                    if ctype is None:
                        # depth 1: the edge's own type, whatever it is. A
                        # non-transitive edge is REACHED here but never
                        # extended below, which is the whole guarantee.
                        new_type = r.relation_type
                    else:
                        new_type = compose(ctype, r.relation_type)
                        if new_type is None:
                            continue
                    new_conf = (r.confidence if ctype is None
                                else compose_confidence(new_type, conf,
                                                        r.confidence))
                    new_path = path + (r,)
                    if r.target == x:
                        continue  # never report the origin as its own closure
                    hit = Derivation(target=r.target, relation_type=new_type,
                                     confidence=new_conf, depth=level,
                                     path=new_path)
                    prior = best.get((r.target, new_type))
                    if prior is None or (
                            (hit.confidence, -hit.depth)
                            > (prior.confidence, -prior.depth)):
                        best[(r.target, new_type)] = hit
                    if new_type not in NON_TRANSITIVE:
                        nxt.append((r.target, new_type, new_conf, new_path))
            frontier = nxt
            if not frontier:
                break
        return sorted(best.values(), key=lambda d: d.sort_key)

    # -- query verb: impact ------------------------------------------------
    def impact(self, x: str, types=None, depth: int = 3) -> list:
        """Reverse-dependency closure: who is affected if ``x`` changes.

        Deliberately a REACHABILITY walk (not a type-composition walk) over
        the reversed dependency edges: a caller of a function inside a file
        that imports a module is genuinely impacted even though
        ``calls ∘ contains`` is not a single named relation. The composed
        type is therefore None and the path is returned as the evidence.
        Confidence is the product along the path — impact at distance is a
        weaker claim than impact next door.
        """
        allowed = self._normalize_types(
            IMPACT_TYPES if types is None else types)
        if depth < 1:
            return []
        best: dict = {}
        frontier = [(x, STATIC_CONFIDENCE, ())]
        visited = {x}
        for level in range(1, depth + 1):
            nxt = []
            for node, conf, path in frontier:
                for r in self.neighbors(node, allowed, direction="in"):
                    src = r.source
                    if src == x:
                        continue
                    new_conf = conf * r.confidence
                    new_path = path + (r,)
                    hit = Derivation(target=src, relation_type=None,
                                     confidence=new_conf, depth=level,
                                     path=new_path)
                    prior = best.get(src)
                    if prior is None or (
                            (hit.confidence, -hit.depth)
                            > (prior.confidence, -prior.depth)):
                        best[src] = hit
                    if src not in visited:
                        visited.add(src)
                        nxt.append((src, new_conf, new_path))
            frontier = nxt
            if not frontier:
                break
        return sorted(best.values(), key=lambda d: d.sort_key)

    # -- query verb: path --------------------------------------------------
    def path(self, x: str, y: str, max_depth: int = 4, types=None,
             direction: str = "out") -> Optional[tuple]:
        """Shortest chain of atoms from ``x`` to ``y``, or None.

        Breadth-first with sorted expansion, so among equally short paths the
        lexicographically first is returned — the same one, every run.
        """
        allowed = self._normalize_types(types)
        if x == y:
            return ()
        frontier = [(x, ())]
        visited = {x}
        for _ in range(max(0, max_depth)):
            nxt = []
            for node, path_so_far in frontier:
                for r in self.neighbors(node, allowed, direction=direction):
                    other = r.target if r.source == node else r.source
                    if other == y:
                        return path_so_far + (r,)
                    if other in visited:
                        continue
                    visited.add(other)
                    nxt.append((other, path_so_far + (r,)))
            frontier = nxt
            if not frontier:
                break
        return None

    # -- query verb: whats_between ----------------------------------------
    def whats_between(self, a: str, b: str, max_depth: int = 4) -> dict:
        """Everything sitting on the routes from ``a`` to ``b``.

        Traversal is undirected (direction='both') because "what is between
        these two things" is a question about connective tissue, not about
        arrow direction. Returns the shortest path, every intermediate node
        found on any simple route within ``max_depth``, and the relation types
        that do the connecting.
        """
        shortest = self.path(a, b, max_depth=max_depth, direction="both")
        routes = self._simple_routes(a, b, max_depth)
        intermediates, types = set(), set()
        for route in routes:
            for r in route:
                types.add(r.relation_type)
                for node in (r.source, r.target):
                    if node not in (a, b):
                        intermediates.add(node)
        return dict(
            connected=shortest is not None,
            shortest_path=[r.as_dict() for r in shortest] if shortest else [],
            hops=len(shortest) if shortest else None,
            intermediates=sorted(intermediates),
            relation_types=sorted(types),
            route_count=len(routes),
        )

    def _simple_routes(self, a: str, b: str, max_depth: int) -> list:
        """All simple undirected routes a->b within max_depth (bounded)."""
        routes = []

        def walk(node, visited, path_so_far):
            if len(path_so_far) >= max_depth:
                return
            for r in self.neighbors(node, None, direction="both"):
                other = r.target if r.source == node else r.source
                if other == b:
                    routes.append(path_so_far + (r,))
                    continue
                if other in visited:
                    continue
                walk(other, visited | {other}, path_so_far + (r,))

        if a != b:
            walk(a, {a}, ())
        return routes

    # -- query verb: shared_structure -------------------------------------
    def shared_structure(self, x: str, y: str) -> dict:
        """Common relation patterns between two nodes — the analogy feed.

        Three kinds of shared structure, kept separate because they mean
        different things: shared TARGETS (both call the same helper), shared
        SOURCES (both are contained by the same file), and a shared type
        SIGNATURE (both are shaped like a definition with N references, even
        with no node in common).
        """
        x_out = {(r.relation_type, r.target)
                 for r in self.neighbors(x, None, "out")}
        y_out = {(r.relation_type, r.target)
                 for r in self.neighbors(y, None, "out")}
        x_in = {(r.relation_type, r.source)
                for r in self.neighbors(x, None, "in")}
        y_in = {(r.relation_type, r.source)
                for r in self.neighbors(y, None, "in")}

        x_sig = self._signature(x)
        y_sig = self._signature(y)
        shared_types = sorted(set(x_sig) & set(y_sig))
        union = set(x_out | y_out | x_in | y_in)
        overlap = (x_out & y_out) | (x_in & y_in)
        return dict(
            shared_targets=sorted(x_out & y_out),
            shared_sources=sorted(x_in & y_in),
            shared_relation_types=shared_types,
            x_signature=x_sig,
            y_signature=y_sig,
            jaccard=(len(overlap) / len(union)) if union else 0.0,
        )

    def _signature(self, node: str) -> dict:
        sig: dict = {}
        for r in self._out.get(node, ()):
            sig[f"out:{r.relation_type}"] = sig.get(
                f"out:{r.relation_type}", 0) + 1
        for r in self._in.get(node, ()):
            sig[f"in:{r.relation_type}"] = sig.get(
                f"in:{r.relation_type}", 0) + 1
        return dict(sorted(sig.items()))

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _normalize_types(types) -> Optional[frozenset]:
        if types is None:
            return None
        if isinstance(types, str):
            types = [types]
        allowed = frozenset(types)
        unknown = allowed - RELATION_TYPES
        if unknown:
            raise RelationTypeError(
                f"unknown relation types in query: {sorted(unknown)}")
        return allowed


# ------------------------------------------------------- silo 1: seams

def seam_node_id(file: str, qualname: str) -> str:
    """The definition ID shared by both silos: ``path::qualname``.

    This is exactly ``chains.seam_manifest``'s seam id, and it is what
    ``relations_from_func_index`` joins onto — the two silos become one graph
    because they agree on this string, not because anything reconciles them.
    """
    return f"{file}::{qualname}"


def relations_from_seam_manifest(manifest: dict) -> list:
    """Adapter: ``chains.seam_manifest`` output -> relation atoms.

    Emits, all at confidence 1.0 (AST-derived, statically verifiable):
      * file  --defines-->     definition        (every definition site)
      * file  --contains-->    definition        (structural containment)
      * class --contains-->    class attribute   (classattr's owner)
      * file  --imports-->     module            (from the imports index)
      * file  --calls-->       definition        (reference kind call/call_attr)
      * file  --references-->  definition        (every other reference kind)

    SHAPE ASSUMPTIONS (read from chains.py, read-only):
      manifest["seams"]      {seam_id: {id, definition, references, ...}}
      definition             {kind, name, qualname, file, lineno[, owner]}
      reference              {kind, name, file, lineno, ...}
      manifest["imports"]    {path: {local: {"module": str, "orig": str|None}}}
      manifest["definitions"] flat list of the definition dicts above
    A manifest missing an optional key degrades to fewer atoms; it never
    raises, because a partial manifest is a normal input (parse errors).
    """
    out = []
    seams = manifest.get("seams") or {}
    for seam_id in sorted(seams):
        seam = seams[seam_id]
        d = seam.get("definition") or {}
        dfile, qual = d.get("file"), d.get("qualname")
        if not dfile or not qual:
            continue
        node = seam.get("id") or seam_node_id(dfile, qual)
        out.append(Relation(dfile, "defines", node,
                            f"seam_manifest:definition:{d.get('kind', '?')}"))
        out.append(Relation(dfile, "contains", node,
                            "seam_manifest:definition"))
        for ref in seam.get("references") or []:
            rfile = ref.get("file")
            if not rfile:
                continue
            kind = ref.get("kind", "?")
            rtype = "calls" if kind in ("call", "call_attr") else "references"
            out.append(Relation(rfile, rtype, node,
                                f"seam_manifest:reference:{kind}"))

    # class attributes: the owning class contains them (a real hierarchy the
    # seam index already knows but only stores as a flat 'owner' string).
    for d in manifest.get("definitions") or []:
        if d.get("kind") != "classattr":
            continue
        dfile, owner, qual = d.get("file"), d.get("owner"), d.get("qualname")
        if not (dfile and owner and qual):
            continue
        out.append(Relation(seam_node_id(dfile, owner), "contains",
                            seam_node_id(dfile, qual),
                            "seam_manifest:classattr"))

    for path in sorted(manifest.get("imports") or {}):
        binds = manifest["imports"][path] or {}
        for local in sorted(binds):
            module = (binds[local] or {}).get("module")
            if module:
                out.append(Relation(path, "imports", module,
                                    "seam_manifest:import"))
    return sorted(out, key=lambda r: r.sort_key)


# --------------------------------------------------- silo 2: function index

def relations_from_func_index(index) -> list:
    """Adapter: ``func_index.build_func_index`` output -> relation atoms.

    Emits, all at confidence 1.0 (AST-derived spans):
      * file     --contains--> func_id     (every record, module records too)
      * file     --defines-->  func_id     (real functions only)
      * parent   --contains--> func_id     (nested / method qualnames)
      * func_id  --part_of-->  seam node   (THE CROSS-SILO JOIN)

    SHAPE ASSUMPTIONS (read from func_index.py, read-only):
      index.records            list[FuncRecord]
      FuncRecord.func_id       "path::qualname::lineno"
      FuncRecord.path/.qualname/.is_module
    Accepts any object exposing ``.records`` (or a bare list of records), so a
    synthetic index needs no FuncIndex construction.

    ``index.adj`` is deliberately NOT emitted as ``imports``: build_index()
    stores import adjacency SYMMETRICALLY (both directions added), so its
    direction is not recoverable. Directional imports come from the seam
    manifest, which records them precisely. Emitting them here would fabricate
    an arrow the source never asserted.
    """
    records = getattr(index, "records", index)
    by_qual = {}
    for rec in records:
        by_qual.setdefault(rec.path, {})[rec.qualname] = rec

    out = []
    for rec in sorted(records, key=lambda r: r.func_id):
        out.append(Relation(rec.path, "contains", rec.func_id,
                            "func_index:record"))
        if rec.is_module:
            continue
        out.append(Relation(rec.path, "defines", rec.func_id,
                            "func_index:function"))
        # the cross-silo join: this span IS the seam manifest's definition
        out.append(Relation(rec.func_id, "part_of",
                            seam_node_id(rec.path, rec.qualname),
                            "func_index:seam_join"))
        if "." in rec.qualname:
            parent_qual = rec.qualname.rsplit(".", 1)[0]
            parent = by_qual.get(rec.path, {}).get(parent_qual)
            if parent is not None:
                out.append(Relation(parent.func_id, "contains", rec.func_id,
                                    "func_index:nesting"))
            else:
                # a method whose owner is a class, not an indexed function:
                # the class only exists in the seam silo, so join to it there.
                out.append(Relation(seam_node_id(rec.path, parent_qual),
                                    "contains", rec.func_id,
                                    "func_index:owner"))
    return sorted(out, key=lambda r: r.sort_key)


def unified_store(manifest: dict = None, index=None) -> RelationStore:
    """Build the v1 unified store from both silos.

    Either silo may be omitted; giving both is the point — the seam adapter
    supplies references/calls/imports against ``path::qualname`` nodes and the
    function-index adapter supplies spans that hang off the same nodes, so a
    single closure crosses what used to be two dictionaries.
    """
    store = RelationStore()
    if manifest:
        store.add_all(relations_from_seam_manifest(manifest))
    if index is not None:
        store.add_all(relations_from_func_index(index))
    return store
