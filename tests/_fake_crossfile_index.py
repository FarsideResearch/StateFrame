"""Deterministic minimal fake of the CrossFileIndex surface (crossfile_index_api).

Hand-rolled over `ast`, per-revision, no I/O: it implements the documented
method names and dataclass fields so chains_recall can be tested before the real
index lands. Scope-aware: a local's type comes from its own function only, from
a constructor call, an annotation, or one hop through a repo callee's return
annotation. Conflicting evidence in one scope binds nothing (fail-open).
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Def:
    path: str
    qualname: str
    kind: str
    lineno: int
    end_lineno: int
    signature: object = None
    returns: str | None = None
    fields: tuple = ()
    bases: tuple = ()      # resolved qualnames ("path::Class") when resolvable, else raw names


@dataclass(frozen=True)
class Ref:
    path: str
    lineno: int
    col: int
    name: str
    kind: str
    receiver: str | None
    target: str | None
    scope: str


def _ann(node) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


class FakeCrossFileIndex:
    def __init__(self, files: dict):
        self._files = dict(files)
        self._trees = {}
        for p in sorted(files):
            try:
                self._trees[p] = ast.parse(files[p])
            except SyntaxError:
                pass
        raw = tuple(sorted(
            (d for p, t in self._trees.items() for d in self._file_defs(p, t)),
            key=lambda d: (d.path, d.lineno, d.qualname)))
        by_name: dict = {}
        for d in raw:
            if d.kind == "class":
                by_name.setdefault(d.qualname, []).append(d)

        def resolve_name(name):     # real-index semantics: unique repo class -> "path::Class"
            hits = by_name.get(name, [])
            return f"{hits[0].path}::{name}" if len(hits) == 1 else name
        self._defs = tuple(
            replace(d, returns=resolve_name(d.returns) if d.returns else None,
                    bases=tuple(resolve_name(b) for b in d.bases)) for d in raw)
        self._classes = {d.qualname: d for d in self._defs if d.kind == "class"}
        self._funcs = {(d.path, d.qualname): d for d in self._defs if d.kind == "def"}
        self._bind_cache = {}

    @classmethod
    def from_sources(cls, sources):
        """{path: str | bytes} -> index (matches the real CrossFileIndex.from_sources)."""
        return cls({p: (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                    for p, v in sources.items()})

    @classmethod
    def build(cls, root, paths):
        """Real-index signature: read `paths` (relative to `root`) from disk."""
        from pathlib import Path
        base = Path(root)
        return cls.from_sources({p: (base / p).read_bytes() for p in paths})

    def update(self, path: str, data: bytes):
        files = dict(self._files)
        files[path] = data.decode("utf-8", "replace")
        return FakeCrossFileIndex(files)

    @property
    def revision(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self._files):
            h.update(p.encode() + hashlib.sha256(self._files[p].encode()).digest())
        return h.hexdigest()[:16]

    # ---- defs
    def _file_defs(self, path, tree):
        out = []
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(Def(path, stmt.name, "def", stmt.lineno, stmt.end_lineno,
                               returns=_ann(stmt.returns)))
            elif isinstance(stmt, ast.ClassDef):
                fields = set()
                for s in stmt.body:
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                        fields.add(s.target.id)
                    elif isinstance(s, ast.Assign):
                        fields |= {t.id for t in s.targets if isinstance(t, ast.Name)}
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                            and n.value.id == "self" and isinstance(n.ctx, ast.Store):
                        fields.add(n.attr)
                out.append(Def(path, stmt.name, "class", stmt.lineno, stmt.end_lineno,
                               fields=tuple(sorted(fields)),
                               bases=tuple(b.id for b in stmt.bases if isinstance(b, ast.Name))))
                for s in stmt.body:
                    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(Def(path, f"{stmt.name}.{s.name}", "def", s.lineno,
                                       s.end_lineno, returns=_ann(s.returns)))
        return out

    def defs(self, path=None):
        return tuple(d for d in self._defs if path is None or d.path == path)

    # ---- bindings
    def _callee_return(self, path, name):
        d = self._funcs.get((path, name))
        if d is None:                                   # imported from a repo module
            for stmt in self._trees[path].body:
                if isinstance(stmt, ast.ImportFrom):
                    for a in stmt.names:
                        if (a.asname or a.name) == name:
                            hits = [x for (p, q), x in sorted(self._funcs.items())
                                    if q == a.name]
                            d = hits[0] if len(hits) == 1 else None
        ret = d.returns.split("::")[-1] if d and d.returns else None
        return ret if ret in self._classes else None

    def _scope_binds(self, path, fn):
        evidence: dict = {}
        declared: dict = {}                             # annotations are authoritative
        def add(name, typ, ann=False):
            (declared if ann else evidence).setdefault(name, set()).add(typ)
        args = [] if fn is None else fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        for a in args:
            if _ann(a.annotation) in self._classes:
                add(a.arg, _ann(a.annotation), True)
        body = self._trees[path] if fn is None else fn
        stack = list(ast.iter_child_nodes(body))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                                # own scope
            stack.extend(ast.iter_child_nodes(n))
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                if _ann(n.annotation) in self._classes:
                    add(n.target.id, _ann(n.annotation), True)
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                    and isinstance(n.value.func, ast.Name):
                f = n.value.func.id
                typ = f if f in self._classes else self._callee_return(path, f)
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        add(t.id, typ)                  # None = unknown evidence
        inferred = {k: next(iter(v)) for k, v in evidence.items()
                    if len(v) == 1 and None not in v}
        return {**inferred, **{k: next(iter(v)) for k, v in declared.items() if len(v) == 1}}

    def binding(self, path, scope, name):
        key = (path, scope)
        if key not in self._bind_cache:
            fn = None
            if scope != "<module>" and path in self._trees:
                fn = next((n for n in ast.walk(self._trees[path])
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.name == scope.split(".")[-1]), None)
            self._bind_cache[key] = self._scope_binds(path, fn) if path in self._trees else {}
        typ = self._bind_cache[key].get(name)
        if typ is None:
            return None
        d = self._classes[typ]
        return f"{d.path}::{d.qualname}"

    # ---- refs
    def refs(self, path):
        tree = self._trees.get(path)
        if tree is None:
            return ()
        out = []
        def walk(node, scope):
            for ch in ast.iter_child_nodes(node):
                sc = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
                if isinstance(ch, ast.Attribute) and isinstance(ch.ctx, ast.Load) \
                        and isinstance(ch.value, ast.Name):
                    out.append(Ref(path, ch.lineno, ch.col_offset, f"{ch.value.id}.{ch.attr}", "attr",
                                   self.binding(path, scope, ch.value.id), None, scope))
                walk(ch, sc)
        walk(tree, "<module>")
        return tuple(sorted(out, key=lambda r: (r.lineno, r.col, r.name)))

    def resolve(self, path, name, lineno):
        if (path, name) in self._funcs or name in self._classes:
            return ("module", name)
        return ("unknown", None)

    # ---- imports
    def import_target(self, path, module, level, name):
        parts = path.split("/")[:-1]
        if level:
            if level - 1 > len(parts):
                return None
            parts = parts[:len(parts) - (level - 1)]
        parts = parts + [p for p in (module or "").split(".") if p]
        stem = "/".join(parts)
        for cand in ((f"{stem}.py", f"{stem}/__init__.py") if stem else ("__init__.py",)):
            if cand in self._trees:
                if name is None:
                    return cand
                if any(d.path == cand and d.qualname == name for d in self._defs):
                    return f"{cand}::{name}"
                for stmt in ast.walk(self._trees[cand]):
                    if isinstance(stmt, ast.ImportFrom) \
                            and any((a.asname or a.name) == name for a in stmt.names):
                        return f"{cand}::{name}"
        return None

    def users(self, qualname):
        return ()


REAL_INDEX_ENV = "SF_REAL_CROSSFILE_INDEX_DIR"


def index_class():
    """The fake by default; crew-02's real CrossFileIndex when REAL_INDEX_ENV names its dir."""
    real_dir = os.environ.get(REAL_INDEX_ENV)
    if not real_dir:
        return FakeCrossFileIndex
    if real_dir not in sys.path:
        sys.path.insert(0, real_dir)
    from kernel_crossfile_index import CrossFileIndex
    return CrossFileIndex
