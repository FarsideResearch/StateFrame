"""SEMANTIC BRIDGE v1 — behavior words -> code locations, without lexical overlap.

The problem (FINDINGS E49-E60): a symptom like "lessons sometimes double-grade"
shares ZERO vocabulary with the buggy line. Every ranker we have is lexical, so
it cannot reach the code. The micro-call verifier is lexical too, which makes the
check circular.

A deterministic bridge cannot map unconstrained prose to code with no anchor at
all. It must attach the behavior to an OBSERVABLE artifact. Per the committed
design (sol_answers/w19/a_semantic_bridge.md) the top two artifacts on a real
FastAPI app are, in the design's own precision/coverage ranking:

  1. ROUTE TABLE       — broadest coverage for HTTP-visible incidents.
                         behavior noun -> route path/tag -> handler -> callees.
  2. STATE-MUTATION MAP — highest PRECISION for correctness bugs (duplication,
                         retries, stale state, idempotency). "double-grade"
                         becomes "who WRITES a *grade* field", which is
                         independent of the buggy line's vocabulary.

Both are built here. Because state-mutation is the higher-precision half, its
hits RANK ABOVE route hits in the fused query (`bridge_candidates`), which is
the ranking the design commits to.

--------------------------------------------------------------------------------
CONSUMPTION CONTRACT (documented, deliberately NOT wired)
--------------------------------------------------------------------------------
`bridge_candidates(symptom_text, repo_root)` returns a deterministically ranked
list of `BridgeCandidate` 4-tuples:

    (path, fn, evidence, bridge_kind)

so a caller can plainly unpack them::

    for path, fn, evidence, kind in bridge_candidates(task_text, root):
        ...

  * `path`      — repo-relative file path, exactly the spelling `func_index`
                  and `chains` use (they share `load_repo_files`).
  * `fn`        — function qualname, the spelling `FuncRecord.qualname` uses.
                  `"<module>"` for a module-level write site.
  * `evidence`  — ONE human-readable line, safe to show a human or ride along in
                  a menu, e.g.
                      "writes LessonProgress.grade at L214 (attribute_assign)"
                      "handles POST /lessons/{id}/grade"
                      "called from handler submit_grade (depth 1)"
  * `bridge_kind` — one of `state_write`, `route_handler`, `route_callee`.
                  Kinds are ordered by the design's precision ranking; the
                  returned list is already in that order.

Intended use in `fixer.py`'s locate stage, per the APPEND-ONLY doctrine:
the existing lexical `function_pool` / `file_menu_from_functions` pool is
computed first and left INTACT; bridge candidates whose `(path, fn)` is already
in the pool contribute their `evidence` line to that entry, and candidates NOT in
the pool are APPENDED as overflow. The bridge never removes, never reorders the
lexical pool's own contents, and never changes behaviour on its own. Evidence
lines are the payload that can ride the menu.

This module is standalone and imports nothing from `fixer.py`, so wiring it is a
one-way dependency whenever the locate pool is ready for it.

--------------------------------------------------------------------------------
Determinism & caching
--------------------------------------------------------------------------------
Every map is a pure function of the corpus text. Results are cached on a hash of
the file contents, so repeated queries against an unchanged tree are free and two
runs over the same tree are byte-identical (ties broken on path/line/name).

Reuse, not reimplementation: identifier splitting comes from
`probes.real_repo_baseline_v2.words`, function spans from
`func_index.build_func_index`, and call edges from `chains.seam_manifest`'s
reference records resolved against those spans.
"""

from __future__ import annotations

import ast
import hashlib
import math
import pathlib
from dataclasses import dataclass
from typing import NamedTuple, Optional

from chains import seam_manifest
from func_index import build_func_index, load_repo_files
from probes.real_repo_baseline_v2 import words

__all__ = [
    "WriteSite",
    "RouteEntry",
    "BridgeCandidate",
    "build_write_map",
    "build_route_map",
    "writers_of",
    "handlers_for",
    "bridge_candidates",
    "noun_tokens",
    "fold_plural",
    "clear_caches",
]

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

# Words that appear in essentially every function in a service codebase. Matching
# on them turns the bridge into "return the whole repo", which is worse than
# returning nothing. Kept deliberately SHORT: over-stopping silently destroys
# recall, and the honest failure mode we want is "no candidates", not "all of
# them". Domain nouns (grade, lesson, progress, ...) are never stopped.
_STOP = frozenset({
    "and", "any", "are", "the", "for", "not", "was", "were", "with", "that",
    "this", "from", "has", "have", "its", "our", "their", "they", "when",
    "get", "set", "put", "add", "new", "old", "run", "use", "used", "make",
    "data", "value", "values", "item", "items", "list", "dict", "key", "keys",
    "name", "names", "type", "types", "file", "files", "line", "lines",
    "code", "codes", "result", "results", "return", "returns", "self",
    "args", "kwargs", "kwarg", "arg", "obj", "res", "req", "tmp", "val",
    "str", "int", "bool", "none", "true", "false",
})


def fold_plural(word: str) -> str:
    """Crude, deterministic plural folding: `grades`->`grade`, `entries`->`entry`.

    Deliberately conservative — it only ever SHORTENS, never invents a stem, so
    it cannot fabricate a match that the raw word would not almost have made.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def noun_tokens(text: str) -> frozenset:
    """Behaviour nouns from free prose OR from an identifier.

    Splits snake_case / camelCase / path segments via the shared `words()` helper
    (so "double-grade", "doubleGrade" and "double_grade" all land on {double,
    grade}), drops boilerplate, folds plurals.
    """
    raw = words(text)
    # words() only keeps [A-Za-z] runs of length >= 3; hyphens already split.
    out = {fold_plural(w) for w in raw if w not in _STOP}
    return frozenset(w for w in out if w and w not in _STOP)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

WRITE_KINDS = (
    "attribute_assign",      # obj.field = ...
    "attribute_augassign",   # obj.field += ...
    "construction",          # Model(field=...)
    "dict_update",           # obj.update({"field": ...}) / .update(field=...)
    "dict_key_assign",       # obj["field"] = ...
    "setattr",               # setattr(obj, "field", ...)
)


@dataclass(frozen=True, order=True)
class WriteSite:
    """One place that writes one field of one entity."""
    path: str
    fn: str
    lineno: int
    write_kind: str

    def evidence(self, entity: str, fld: str) -> str:
        return f"writes {entity}.{fld} at L{self.lineno} ({self.write_kind})"


@dataclass(frozen=True, order=True)
class RouteEntry:
    """One HTTP route and the handler it dispatches to."""
    path: str            # file that declares the route
    route: str           # full path incl. router prefix, e.g. /lessons/{id}/grade
    method: str          # GET/POST/... (upper), or "ANY"
    handler: str         # handler function qualname
    lineno: int
    tags: tuple = ()

    def evidence(self) -> str:
        return f"handles {self.method} {self.route}"


class BridgeCandidate(NamedTuple):
    """The consumption contract. Exactly four fields, so callers can unpack."""
    path: str
    fn: str
    evidence: str
    bridge_kind: str


# Precision ranking from the design: state-mutation hits outrank route hits.
_KIND_RANK = {"state_write": 0, "route_handler": 1, "route_callee": 2}


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

_WRITE_CACHE: dict = {}
_ROUTE_CACHE: dict = {}
_FILES_CACHE: dict = {}
_TOKW_CACHE: dict = {}


def clear_caches() -> None:
    _WRITE_CACHE.clear()
    _ROUTE_CACHE.clear()
    _FILES_CACHE.clear()
    _TOKW_CACHE.clear()


def _files_hash(files: dict) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode())
        h.update(b"\0")
        h.update(files[p].encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _resolve_files(repo_root, files: Optional[dict]) -> dict:
    """Accept either an explicit {path: src} dict or a repo root to load."""
    if files is not None:
        return files
    key = str(repo_root)
    if key not in _FILES_CACHE:
        _FILES_CACHE[key] = load_repo_files(pathlib.Path(repo_root))
    return _FILES_CACHE[key]


# ---------------------------------------------------------------------------
# BRIDGE 1 — state-mutation map
# ---------------------------------------------------------------------------


class _WriteScan(ast.NodeVisitor):
    """Collect (entity, field) -> write sites for one file.

    Entity resolution, best-effort and always fail-open (an unresolved receiver
    is recorded under its own variable name rather than dropped):

      * inside a method of class C, `self.f = ...`      -> entity C
      * `v = Model(...)` anywhere                        -> v bound to Model
      * `v: Model` / `def g(v: Model)`                   -> v bound to Model
      * otherwise the receiver's variable name is the entity

    A name bound to two different classes is ambiguous and is dropped rather than
    guessed (same rule `chains._FileScan.finish` uses).
    """

    def __init__(self, path: str):
        self.path = path
        self.writes: list = []          # (entity, field, WriteSite)
        self._binds: dict = {}
        self._classes: list = []        # enclosing ClassDef name stack
        self._fns: list = []            # enclosing function qualname stack
        self._known_classes: set = set()

    # -- binding pass helpers -------------------------------------------------
    def _bind(self, name, cls):
        self._binds.setdefault(name, set()).add(cls)

    def resolved(self) -> dict:
        return {n: next(iter(c)) for n, c in self._binds.items() if len(c) == 1}

    @property
    def _fn(self) -> str:
        return self._fns[-1] if self._fns else "<module>"

    # -- scope tracking -------------------------------------------------------
    def visit_ClassDef(self, node):
        # Class names join the qualname stack so a method's `fn` reads
        # `Class.method` — the same spelling `func_index` uses.
        self._known_classes.add(node.name)
        self._classes.append(node.name)
        self._fns.append(f"{self._fns[-1]}.{node.name}" if self._fns else node.name)
        self.generic_visit(node)
        self._fns.pop()
        self._classes.pop()

    def _visit_fn(self, node):
        qual = f"{self._fn}.{node.name}" if self._fns else node.name
        self._fns.append(qual)
        for a in list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs):
            if isinstance(a.annotation, ast.Name):
                self._bind(a.arg, a.annotation.id)
        self.generic_visit(node)
        self._fns.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    # -- writes ---------------------------------------------------------------
    def _record(self, entity, fld, lineno, kind):
        if not entity or not fld:
            return
        self.writes.append((entity, fld,
                            WriteSite(self.path, self._fn, lineno, kind)))

    def _receiver(self, node) -> Optional[str]:
        """Name of the receiver expression, or None if it is not a plain name."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # obj.sub.field = ... -> attribute the write to `sub`
            return node.attr
        return None

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._bind(t.id, node.value.func.id)
        for t in node.targets:
            if isinstance(t, ast.Attribute):
                self._record(self._receiver(t.value), t.attr,
                             node.lineno, "attribute_assign")
            elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) \
                    and isinstance(t.slice.value, str):
                self._record(self._receiver(t.value), t.slice.value,
                             node.lineno, "dict_key_assign")
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and isinstance(node.annotation, ast.Name):
            self._bind(node.target.id, node.annotation.id)
        if isinstance(node.target, ast.Attribute):
            self._record(self._receiver(node.target.value), node.target.attr,
                         node.lineno, "attribute_assign")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Attribute):
            self._record(self._receiver(node.target.value), node.target.attr,
                         node.lineno, "attribute_augassign")
        elif isinstance(node.target, ast.Subscript) \
                and isinstance(node.target.slice, ast.Constant) \
                and isinstance(node.target.slice.value, str):
            self._record(self._receiver(node.target.value), node.target.slice.value,
                         node.lineno, "attribute_augassign")
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        # Model(field=...) — ORM-style construction. Only for names that look
        # like a class (CapWords); a lowercase factory call is not an entity.
        if isinstance(f, ast.Name) and _looks_like_class(f.id):
            for kw in node.keywords:
                if kw.arg:
                    self._record(f.id, kw.arg, node.lineno, "construction")
        # obj.update({"field": v}) / obj.update(field=v)
        elif isinstance(f, ast.Attribute) and f.attr == "update":
            base = self._receiver(f.value)
            for kw in node.keywords:
                if kw.arg:
                    self._record(base, kw.arg, node.lineno, "dict_update")
            for a in node.args:
                if isinstance(a, ast.Dict):
                    for k in a.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            self._record(base, k.value, node.lineno, "dict_update")
                        elif isinstance(k, ast.Attribute):
                            # SQLAlchemy .update({Model.field: v})
                            self._record(self._receiver(k.value), k.attr,
                                         node.lineno, "dict_update")
        # setattr(obj, "field", v)
        elif isinstance(f, ast.Name) and f.id == "setattr" and len(node.args) >= 2:
            tgt, name = node.args[0], node.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                self._record(self._receiver(tgt), name.value,
                             node.lineno, "setattr")
        self.generic_visit(node)


def _looks_like_class(name: str) -> bool:
    return bool(name) and name[0].isupper() and not name.isupper()


def build_write_map(repo_root=None, files: Optional[dict] = None) -> dict:
    """{(entity, field): [WriteSite, ...]} over the corpus.

    Deterministic and cached on the corpus content hash. Each value list is
    sorted by (path, lineno, fn) so two runs agree byte-for-byte.
    """
    files = _resolve_files(repo_root, files)
    key = _files_hash(files)
    if key in _WRITE_CACHE:
        return _WRITE_CACHE[key]

    out: dict = {}
    for path in sorted(files):
        try:
            tree = ast.parse(files[path])
        except SyntaxError:
            continue  # a file we cannot parse contributes nothing; never raises
        scan = _WriteScan(path)
        scan.visit(tree)
        binds = scan.resolved()
        for entity, fld, site in scan.writes:
            # normalize the receiver to a class name where we can derive one
            ent = binds.get(entity, entity)
            if ent == "self":
                # attribute the write to the enclosing class if we know it
                ent = _enclosing_class(site.fn) or "self"
            out.setdefault((ent, fld), []).append(site)

    for k in out:
        out[k] = sorted(set(out[k]))
    _WRITE_CACHE[key] = out
    return out


def _enclosing_class(qualname: str) -> Optional[str]:
    """`Foo.bar` -> `Foo`; a bare function has no class."""
    parts = qualname.split(".")
    return parts[-2] if len(parts) >= 2 else None


def _token_weights(vocab_counts: dict) -> dict:
    """Inverse-frequency weight per token: a noun that matches half the corpus is
    not evidence. Purely a function of the map, so it stays deterministic.

    Without this, a symptom like "lessons double-grade" ranks every writer of a
    `lesson_id` field above the writers of a `grade` field, because `lesson` is
    everywhere and `grade` is rare — exactly backwards.
    """
    return {t: 1.0 / (1.0 + math.log(1.0 + c)) for t, c in vocab_counts.items()}


def _write_map_token_weights(write_map: dict) -> dict:
    key = id(write_map)
    cached = _TOKW_CACHE.get(key)
    if cached is not None and cached[0] is write_map:
        return cached[1]
    counts: dict = {}
    for (entity, fld) in write_map:
        for t in noun_tokens(fld) | noun_tokens(entity):
            counts[t] = counts.get(t, 0) + 1
    w = _token_weights(counts)
    _TOKW_CACHE[key] = (write_map, w)
    return w


def writers_of(behavior_nouns, write_map: dict) -> list:
    """Rank (entity, field, sites) by how well the FIELD matches the nouns.

    Field agreement is the strong signal ("double-grade" -> writers of *grade*
    fields), entity agreement is a weaker corroborating one, and every matched
    token is weighted by how rare it is in the map. Returns
    [(score, entity, field, [WriteSite,...])] sorted deterministically.
    """
    nouns = behavior_nouns if isinstance(behavior_nouns, (set, frozenset)) \
        else noun_tokens(" ".join(behavior_nouns) if not isinstance(behavior_nouns, str)
                         else behavior_nouns)
    nouns = frozenset(fold_plural(n) for n in nouns)
    if not nouns:
        return []

    tw = _write_map_token_weights(write_map)
    hits = []
    for (entity, fld), sites in write_map.items():
        ftok = noun_tokens(fld)
        etok = noun_tokens(entity)
        fmatch = ftok & nouns
        ematch = (etok & nouns) - fmatch
        if not fmatch and not ematch:
            continue
        # field match dominates; entity match alone is a weak, still-reported hit
        score = (2.0 * sum(tw.get(t, 1.0) for t in fmatch)
                 + 1.0 * sum(tw.get(t, 1.0) for t in ematch))
        if not fmatch:
            score *= 0.4
        hits.append((round(score, 6), entity, fld, sites))
    hits.sort(key=lambda h: (-h[0], h[1], h[2]))
    return hits


# ---------------------------------------------------------------------------
# BRIDGE 2 — route table
# ---------------------------------------------------------------------------

_HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head",
                           "options", "trace"})


def _str_const(node) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _str_list(node) -> tuple:
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(v for v in (_str_const(e) for e in node.elts) if v)
    return ()


class _RouteScan(ast.NodeVisitor):
    """Routes declared in one file: `@router.get("/x")` on a def, plus the
    module-level `APIRouter(prefix=..., tags=[...])` those decorators hang off."""

    def __init__(self, path: str):
        self.path = path
        self.routers: dict = {}     # local name -> {"prefix":..., "tags": (...)}
        self.routes: list = []
        self._fns: list = []

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            fn = node.value.func
            fname = fn.id if isinstance(fn, ast.Name) else \
                (fn.attr if isinstance(fn, ast.Attribute) else "")
            if fname in ("APIRouter", "FastAPI"):
                prefix, tags = "", ()
                for kw in node.value.keywords:
                    if kw.arg == "prefix":
                        prefix = _str_const(kw.value) or ""
                    elif kw.arg == "tags":
                        tags = _str_list(kw.value)
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.routers[t.id] = {"prefix": prefix, "tags": tags}
        self.generic_visit(node)

    def _visit_fn(self, node):
        qual = f"{self._fns[-1]}.{node.name}" if self._fns else node.name
        for dec in node.decorator_list:
            self._route_from_decorator(dec, qual, node.lineno)
        self._fns.append(qual)
        self.generic_visit(node)
        self._fns.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    def visit_ClassDef(self, node):
        self._fns.append(f"{self._fns[-1]}.{node.name}" if self._fns else node.name)
        self.generic_visit(node)
        self._fns.pop()

    def _route_from_decorator(self, dec, handler_qual, lineno):
        if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
            return
        attr = dec.func.attr
        base = dec.func.value.id if isinstance(dec.func.value, ast.Name) else None
        if base is None:
            return
        if attr in _HTTP_METHODS:
            methods = (attr.upper(),)
        elif attr == "api_route":
            methods = tuple(m.upper() for kw in dec.keywords if kw.arg == "methods"
                            for m in _str_list(kw.value)) or ("ANY",)
        else:
            return
        raw = _str_const(dec.args[0]) if dec.args else None
        if raw is None:
            return
        meta = self.routers.get(base, {"prefix": "", "tags": ()})
        dec_tags = ()
        for kw in dec.keywords:
            if kw.arg == "tags":
                dec_tags = _str_list(kw.value)
        route = (meta["prefix"] + raw) or raw
        tags = tuple(sorted(set(meta["tags"]) | set(dec_tags)))
        for m in methods:
            self.routes.append(RouteEntry(path=self.path, route=route, method=m,
                                          handler=handler_qual, lineno=lineno,
                                          tags=tags))


def _span_owner(fidx, path: str) -> list:
    """(lineno, end_lineno, qualname) spans for one file, innermost last."""
    recs = [r for r in fidx.records if r.path == path and not r.is_module]
    return sorted(((r.lineno, r.end_lineno, r.qualname) for r in recs),
                  key=lambda s: (s[0], -(s[1] - s[0])))


def _fn_at(spans, lineno: int) -> Optional[str]:
    """Innermost function containing `lineno`."""
    best = None
    for lo, hi, q in spans:
        if lo <= lineno <= hi:
            if best is None or (hi - lo) <= best[0]:
                best = (hi - lo, q)
    return best[1] if best else None


def build_route_map(repo_root=None, files: Optional[dict] = None,
                    max_depth: int = 2) -> dict:
    """Route table + bounded transitive callees per handler.

    Returns::

        {"routes":  [RouteEntry, ...],
         "callees": {(path, handler): [(depth, path, fn), ...]},
         "max_depth": int}

    Call edges are NOT reimplemented: they are `chains.seam_manifest`'s call /
    call_attr reference records (which carry file+lineno) resolved to a caller
    via `func_index` function spans, and to a callee via the manifest's own
    definition index. Depth is bounded by `max_depth` and every visited node is
    remembered, so a cycle terminates.
    """
    files = _resolve_files(repo_root, files)
    key = (_files_hash(files), max_depth)
    if key in _ROUTE_CACHE:
        return _ROUTE_CACHE[key]

    routes: list = []
    for path in sorted(files):
        try:
            tree = ast.parse(files[path])
        except SyntaxError:
            continue
        scan = _RouteScan(path)
        scan.visit(tree)
        routes.extend(scan.routes)
    routes.sort()

    manifest = seam_manifest(files)
    fidx = build_func_index(files)
    spans = {p: _span_owner(fidx, p) for p in sorted(files)}

    # caller -> set of callee names, from the manifest's call reference records
    call_names: dict = {}
    for ref in manifest["references"]:
        if ref["kind"] not in ("call", "call_attr"):
            continue
        caller = _fn_at(spans.get(ref["file"], []), ref["lineno"])
        if not caller:
            continue
        call_names.setdefault((ref["file"], caller), set()).add(ref["name"])

    # callee name -> its definition sites (path, qualname)
    defs_by_name: dict = {}
    for d in manifest["definitions"]:
        if d["kind"] in ("function", "classattr"):
            defs_by_name.setdefault(d["name"], set()).add((d["file"], d["qualname"]))

    callees: dict = {}
    for r in routes:
        start = (r.path, r.handler)
        seen = {start}
        frontier = [start]
        found: list = []
        for depth in range(1, max_depth + 1):
            nxt = []
            for node in frontier:
                for nm in sorted(call_names.get(node, ())):
                    for tgt in sorted(defs_by_name.get(nm, ())):
                        if tgt in seen:
                            continue
                        seen.add(tgt)
                        found.append((depth, tgt[0], tgt[1]))
                        nxt.append(tgt)
            frontier = nxt
            if not frontier:
                break
        callees[start] = sorted(found)

    out = {"routes": routes, "callees": callees, "max_depth": max_depth}
    _ROUTE_CACHE[key] = out
    return out


def handlers_for(behavior_phrase, route_map: dict) -> list:
    """Rank routes whose PATH SEGMENTS, TAGS or HANDLER NAME match the nouns.

    Returns [(score, RouteEntry)] sorted deterministically.
    """
    nouns = behavior_phrase if isinstance(behavior_phrase, (set, frozenset)) \
        else noun_tokens(behavior_phrase)
    nouns = frozenset(fold_plural(n) for n in nouns)
    if not nouns:
        return []
    counts: dict = {}
    for r in route_map["routes"]:
        for t in noun_tokens(r.route) | noun_tokens(" ".join(r.tags)) \
                | noun_tokens(r.handler):
            counts[t] = counts.get(t, 0) + 1
    tw = _token_weights(counts)
    hits = []
    for r in route_map["routes"]:
        # a path param {lesson_id} is a real referent, so tokenise the whole route
        route_tok = noun_tokens(r.route)
        tag_tok = noun_tokens(" ".join(r.tags))
        name_tok = noun_tokens(r.handler)
        m_route, m_tag, m_name = route_tok & nouns, tag_tok & nouns, name_tok & nouns
        if not (m_route or m_tag or m_name):
            continue
        score = round(2.0 * sum(tw.get(t, 1.0) for t in m_route)
                      + 1.5 * sum(tw.get(t, 1.0) for t in m_tag)
                      + 1.0 * sum(tw.get(t, 1.0) for t in m_name), 6)
        hits.append((score, r))
    hits.sort(key=lambda h: (-h[0], h[1]))
    return hits


# ---------------------------------------------------------------------------
# THE BRIDGE QUERY
# ---------------------------------------------------------------------------

# Bases keep the design's precision ranking intact: any state-mutation hit
# outranks any route hit, and a handler outranks its own callees.
_BASE = {"state_write": 1000.0, "route_handler": 500.0, "route_callee": 250.0}


def bridge_candidates(symptom_text: str, repo_root=None, *,
                      files: Optional[dict] = None,
                      limit: int = 25,
                      max_depth: int = 2,
                      interleave: bool = False) -> list:
    """Fuse both bridges into one ranked candidate list.

    See the module docstring for the consumption contract. Deterministic: equal
    scores are broken on (kind rank, path, fn).

    `interleave=False` (the default) is the design's committed strict precision
    ordering: every state-mutation hit outranks every route hit. MEASURED
    CAVEAT: on this corpus that ordering starves the route bridge completely —
    across the 31 messy front-door asks the top-5 was 155/155 `state_write`, and
    a route candidate first appears only around rank 20. `interleave=True`
    round-robins the three kinds (still precision-ordered within each kind and
    still fully deterministic) so a route handler can reach the top-5. It is
    OFF by default so the default behaviour matches the committed design.
    """
    files = _resolve_files(repo_root, files)
    nouns = noun_tokens(symptom_text)
    if not nouns:
        return []

    write_map = build_write_map(files=files)
    route_map = build_route_map(files=files, max_depth=max_depth)

    # (path, fn) -> [score, kind, evidence]
    best: dict = {}

    def offer(path, fn, score, kind, evidence):
        cur = best.get((path, fn))
        if cur is None or (score, -_KIND_RANK[kind]) > (cur[0], -_KIND_RANK[cur[1]]):
            best[(path, fn)] = [score, kind, evidence]

    # -- bridge 1: state mutation (highest precision -> ranks first)
    for score, entity, fld, sites in writers_of(nouns, write_map):
        for s in sites:
            offer(s.path, s.fn, _BASE["state_write"] + score,
                  "state_write", s.evidence(entity, fld))

    # -- bridge 2: routes, then their bounded callees
    for score, r in handlers_for(nouns, route_map):
        offer(r.path, r.handler, _BASE["route_handler"] + score,
              "route_handler", r.evidence())
        for depth, cpath, cfn in route_map["callees"].get((r.path, r.handler), ()):
            offer(cpath, cfn,
                  _BASE["route_callee"] + score - depth,
                  "route_callee",
                  f"called from handler {r.handler} (depth {depth}) "
                  f"for {r.method} {r.route}")

    ranked = sorted(
        (BridgeCandidate(p, f, ev, kind), sc)
        for (p, f), (sc, kind, ev) in best.items()
    )
    ranked.sort(key=lambda t: (-t[1], _KIND_RANK[t[0].bridge_kind],
                               t[0].path, t[0].fn))
    if not interleave:
        return [c for c, _ in ranked[:limit]]

    lanes = {k: [c for c, _ in ranked if c.bridge_kind == k]
             for k in ("state_write", "route_handler", "route_callee")}
    out: list = []
    i = 0
    while len(out) < limit and any(len(v) > i for v in lanes.values()):
        for k in ("state_write", "route_handler", "route_callee"):
            if len(lanes[k]) > i and len(out) < limit:
                out.append(lanes[k][i])
        i += 1
    return out
