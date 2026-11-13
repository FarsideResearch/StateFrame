"""Per-revision cross-file index of a Python repo (read-only, deterministic).

Prior art, reused vs added:
  * symbol_registry: reused its signature rendering rule (keep the / and * markers; a
    subtly wrong signature is worse than none) as structured Param records instead of text.
  * kernel_reference_index: reused the content-digest cache idea and the skip-dir set.
  * kernel_workspace_truth: reused its "unknown is quiet, never a false absent" stance;
    added what it lacks (lowercase module vars, re-exports) and parses each file once.
  * mc_repo_intel_vendor/call_graph + kernel_repo_intel: function-level call graph with an
    optional tree-sitter/LSP stack; too heavy and name-keyed here, so not wrapped.
  * func_index: per-function digests for ranking; no imports or refs, so not reused.
  * Added: scope-aware resolution, imports with re-export following, refs with receiver
    types, one-hop bindings, reverse edges and dangling detection in one immutable index.
Nothing in the engine calls this module yet; STATEFRAME_CROSSFILE_INDEX defaults to off.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kernel_crossfile_extract import (
    MODULE_SCOPE, Def, FileFacts, Import, Param, RawRef, Scope, Signature, chain, extract_facts,
)

CROSSFILE_INDEX_ENV = "STATEFRAME_CROSSFILE_INDEX"
MAX_FILE_BYTES = 1_000_000   # over this a file is indexed as defs and imports only
MAX_HARD_BYTES = 8_000_000   # over this a file is not read at all
MAX_FILES = 20_000
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "node_modules", "dist", "build", "site-packages",
})
_MAX_DEPTH = 16
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))
_BUILTINS = frozenset(dir(builtins))


def enabled() -> bool:
    """Read the switch; junk values raise so a typo cannot silently run in the wrong mode."""
    raw = os.environ.get(CROSSFILE_INDEX_ENV, "off").strip().lower()
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise ValueError(f"{CROSSFILE_INDEX_ENV} must be 'on' or 'off', got {raw!r}")


@dataclass(frozen=True, slots=True)
class Ref:
    path: str
    lineno: int
    col: int
    name: str
    kind: str  # call | attr | name | import | key
    receiver: str | None
    target: str | None
    scope: str


@dataclass(frozen=True, slots=True)
class Resolution:
    kind: str  # local | param | module | import | builtin | unknown
    target: str | None = None


@dataclass(frozen=True, slots=True)
class CallSite:
    path: str
    lineno: int
    nargs: int
    kwnames: tuple[str, ...]
    has_star: bool


@dataclass(frozen=True, slots=True)
class ParseError:
    path: str
    reason: str  # parse_error | not_utf8 | too_large | max_files | unreadable


@dataclass(frozen=True, slots=True)
class _Res:
    status: str  # def | module | external | unknown | missing
    target: str = ""   # module target (file path or "dir:<prefix>")
    qual: str = ""     # "path::qualname" for def
    reason: str = ""


_UNKNOWN = _Res("unknown")
_EXTERNAL = _Res("external")


@dataclass(frozen=True, slots=True)
class _Entry:
    sha: str
    facts: FileFacts | None
    error: str = ""
    defs_only: bool = False


class _Shared:
    """Add-only parse cache shared by every revision derived from one build (keyed by sha)."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, bool], FileFacts] = {}
        self.parses = 0


def _entry(shared: _Shared, raw: bytes, max_file_bytes: int) -> _Entry:
    sha = hashlib.sha256(raw).hexdigest()
    defs_only = len(raw) > max_file_bytes
    key = (sha, defs_only)
    facts = shared.cache.get(key)
    if facts is None:
        try:
            facts = extract_facts(raw.decode("utf-8"), defs_only=defs_only)
        except UnicodeDecodeError:
            return _Entry(sha, None, "not_utf8")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return _Entry(sha, None, "parse_error")
        shared.parses += 1
        shared.cache[key] = facts
    return _Entry(sha, facts, "", defs_only)


def _norm(path: str | Path) -> str:
    text = Path(path).as_posix()
    while text.startswith("./"):
        text = text[2:]
    return text


def _type_head(text: str | None, depth: int = 0) -> str | None:
    """Dotted class name inside an annotation: unwraps quotes, Optional, Union[..., None], X | None."""
    if not text or depth > 4:
        return None
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    return _type_node(node, depth)


def _type_node(node: ast.AST, depth: int) -> str | None:
    if isinstance(node, ast.Constant):
        return _type_head(node.value, depth + 1) if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _type_node(node.left, depth) or _type_node(node.right, depth)
    if isinstance(node, ast.Subscript):
        outer = chain(node.value)
        if outer is not None and outer[0].rsplit(".", 1)[-1] in ("Optional", "Union", "Annotated"):
            elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for e in elts:
                got = _type_node(e, depth)
                if got and got != "None":
                    return got
        return None
    c = chain(node)
    return c[0] if c and c[0] != "None" else None


class CrossFileIndex:
    """Immutable per-revision index; update() returns a new index sharing the parse cache."""

    def __init__(self, entries: Mapping[str, _Entry], shared: _Shared, max_file_bytes: int, max_files: int) -> None:
        self._entries = dict(entries)
        self._shared = shared
        self._max_file_bytes = max_file_bytes
        self._max_files = max_files
        self._dirs = {"/".join(p.split("/")[:i]) for p in self._entries for i in range(1, p.count("/") + 1)}
        self._raw: dict[str, dict[str, Def]] = {}
        self._scopes_by_path: dict[str, dict[str, Scope]] = {}
        self._memo: dict[tuple, object] = {}
        self._refs: dict[str, tuple[Ref, ...]] = {}
        self._users: dict[str, list[Ref]] | None = None
        self._calls: dict[str, list[CallSite]] | None = None
        self.revision = hashlib.sha256("".join(
            f"{p}\0{self._entries[p].sha}\n" for p in sorted(self._entries)).encode()).hexdigest()

    # ---- construction
    @classmethod
    def build(cls, root: str | Path, paths: Sequence[str] | None = None, *,
              max_files: int = MAX_FILES, max_file_bytes: int = MAX_FILE_BYTES) -> "CrossFileIndex":
        """Index `paths` (relative to root) or, when omitted, every .py file under root."""
        base = Path(root).resolve()
        wanted = cls._scan(base) if paths is None else sorted({_norm(p) for p in paths})
        shared = _Shared()
        entries: dict[str, _Entry] = {}
        for rel in [p for p in wanted if p.endswith(".py")]:
            if len(entries) >= max_files:
                entries[rel] = _Entry("max_files", None, "max_files")
                continue
            entries[rel] = cls._read(shared, base / rel, base, max_file_bytes)
        return cls(entries, shared, max_file_bytes, max_files)

    @staticmethod
    def _scan(base: Path) -> list[str]:
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            found += [Path(dirpath, n).relative_to(base).as_posix() for n in sorted(filenames) if n.endswith(".py")]
        return found

    @staticmethod
    def _read(shared: _Shared, full: Path, base: Path, max_file_bytes: int) -> _Entry:
        try:
            if full.is_symlink() and not full.resolve().is_relative_to(base):
                return _Entry("unreadable", None, "unreadable")
            size = full.stat().st_size
            if size > MAX_HARD_BYTES:
                return _Entry(f"too_large:{size}", None, "too_large")
            return _entry(shared, full.read_bytes(), max_file_bytes)
        except OSError:
            return _Entry("unreadable", None, "unreadable")

    @classmethod
    def from_sources(cls, sources: Mapping[str, str | bytes], *, max_file_bytes: int = MAX_FILE_BYTES,
                     max_files: int = MAX_FILES) -> "CrossFileIndex":
        """In-memory build (same result as build() over files with these contents)."""
        shared = _Shared()
        entries: dict[str, _Entry] = {}
        for rel in sorted({_norm(p) for p in sources}):
            raw = next(v for k, v in sources.items() if _norm(k) == rel)
            raw = raw.encode("utf-8") if isinstance(raw, str) else raw
            if not rel.endswith(".py"):
                continue
            if len(entries) >= max_files:
                entries[rel] = _Entry("max_files", None, "max_files")
            elif len(raw) > MAX_HARD_BYTES:
                entries[rel] = _Entry(f"too_large:{len(raw)}", None, "too_large")
            else:
                entries[rel] = _entry(shared, raw, max_file_bytes)
        return cls(entries, shared, max_file_bytes, max_files)

    def update(self, path: str, data: bytes | None) -> "CrossFileIndex":
        """New index with `path` reparsed from `data` (None removes it). Unchanged bytes reuse the cache."""
        rel = _norm(path)
        entries = dict(self._entries)
        if data is None:
            entries.pop(rel, None)
        elif rel.endswith(".py"):
            if len(data) > MAX_HARD_BYTES:
                entries[rel] = _Entry(f"too_large:{len(data)}", None, "too_large")
            else:
                entries[rel] = _entry(self._shared, data, self._max_file_bytes)
        return CrossFileIndex(entries, self._shared, self._max_file_bytes, self._max_files)

    # ---- inventory
    @property
    def parses(self) -> int:
        return self._shared.parses

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def parse_errors(self) -> tuple[ParseError, ...]:
        return tuple(ParseError(p, e.error) for p, e in sorted(self._entries.items()) if e.error)

    def defs_only_files(self) -> tuple[str, ...]:
        return tuple(p for p, e in sorted(self._entries.items()) if e.defs_only)

    def _facts(self, path: str) -> FileFacts | None:
        e = self._entries.get(path)
        return e.facts if e else None

    def _rawdefs(self, path: str) -> dict[str, Def]:
        got = self._raw.get(path)
        if got is None:
            f = self._facts(path)
            got = {d.qualname: replace(d, path=path) for d in f.defs} if f else {}
            self._raw[path] = got
        return got

    def _scopes(self, path: str) -> dict[str, Scope]:
        got = self._scopes_by_path.get(path)
        if got is None:
            f = self._facts(path)
            got = {s.qualname: s for s in f.scopes} if f else {}
            self._scopes_by_path[path] = got
        return got

    def _bindings(self, path: str) -> dict[str, Import]:
        f = self._facts(path)
        return {i.local: i for i in f.imports if i.kind != "star"} if f else {}

    def _def(self, qual: str) -> Def | None:
        path, _, q = qual.partition("::")
        return self._rawdefs(path).get(q)

    # ---- module resolution
    def _find(self, parts: list[str]) -> str | None:
        path = "/".join(parts)
        if path:
            if f"{path}.py" in self._entries:
                return f"{path}.py"
            if f"{path}/__init__.py" in self._entries:
                return f"{path}/__init__.py"
        elif "__init__.py" in self._entries:
            return "__init__.py"
        return f"dir:{path}" if path in self._dirs or not path else None

    def _module_target(self, dotted: str, from_file: str, level: int) -> tuple[str | None, bool]:
        """(repo target or None, anchored). Unanchored means stdlib/third-party: never flagged."""
        parts = dotted.split(".") if dotted else []
        here = from_file.split("/")[:-1]
        if level:
            if level - 1 > len(here):
                return None, True
            return self._find([*here[: len(here) - (level - 1)], *parts]), True
        if parts and parts[0] in _STDLIB and self._find(parts[:1]) is None:
            return None, False
        seen: list[list[str]] = []
        anchored = False
        for b in ([], ["src"], here):
            if b in seen:
                continue
            seen.append(b)
            anchored = anchored or (bool(parts) and self._find([*b, parts[0]]) is not None)
            hit = self._find([*b, *parts])
            if hit is not None:
                return hit, True
        return None, anchored

    def _submodule(self, pkg_target: str, name: str) -> str | None:
        if pkg_target.startswith("dir:"):
            prefix = pkg_target[4:].split("/") if pkg_target[4:] else []
        elif pkg_target.rsplit("/", 1)[-1] == "__init__.py":
            prefix = pkg_target.split("/")[:-1]
        else:
            return None
        return self._find([*prefix, name])

    # ---- import / attribute resolution
    def _import_res(self, path: str, imp: Import, depth: int = 0) -> _Res:
        key = ("imp", path, imp.line, imp.local, imp.module, imp.name, imp.level)
        hit = self._memo.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        self._memo[key] = _UNKNOWN  # cycle guard
        if imp.kind == "star":
            res = _UNKNOWN
        else:
            tgt, anchored = self._module_target(imp.module, path, imp.level)
            if tgt is None:
                res = _Res("missing", reason="module_not_found") if anchored else _EXTERNAL
            elif imp.kind == "import":
                bound = tgt if imp.bound_module == imp.module else self._module_target(imp.bound_module, path, 0)[0]
                res = _Res("module", target=bound or tgt)
            else:
                res = self._attr(tgt, [imp.name], depth + 1)
                if res.status == "missing":
                    res = _Res("missing", reason="name_not_defined")
        self._memo[key] = res
        return res

    def _attr(self, target: str, chain_: list[str], depth: int) -> _Res:
        if not chain_:
            return _Res("module", target=target)
        if depth > _MAX_DEPTH:
            return _UNKNOWN
        first = chain_[0]
        if not target.startswith("dir:"):
            if self._facts(target) is None:
                return _UNKNOWN  # unparseable file: contents unknown, never a false absent
            defs = self._rawdefs(target)
            for n in range(len(chain_), 0, -1):  # longest defined prefix wins
                d = defs.get(".".join(chain_[:n]))
                if d is not None:
                    return _Res("def", qual=d.qualified)
            imp = self._bindings(target).get(first)
            if imp is not None:
                r = self._import_res(target, imp, depth)
                if r.status != "unknown":  # unknown here may be `from . import x` pointing at itself
                    return self._descend(r, chain_[1:], depth)
        sub = self._submodule(target, first)
        if sub is not None:
            return self._attr(sub, chain_[1:], depth + 1)
        if not target.startswith("dir:"):
            stars = [i for i in self._facts(target).imports if i.kind == "star"]  # type: ignore[union-attr]
            if stars or "__getattr__" in self._rawdefs(target):
                for s in stars:
                    tgt, _ = self._module_target(s.module, target, s.level)
                    if tgt is not None:
                        r = self._attr(tgt, chain_, depth + 1)
                        if r.status in ("def", "module"):
                            return r
                return _UNKNOWN
        return _Res("missing", reason="attribute_not_defined")

    def _descend(self, r: _Res, rest: list[str], depth: int) -> _Res:
        if r.status == "module":
            return self._attr(r.target, rest, depth + 1) if rest else r
        return r

    # ---- scope-aware lookup
    def _lookup(self, path: str, scope: str, head: str) -> str:
        scopes = self._scopes(path)
        q = scope
        while q and q != MODULE_SCOPE:
            s = scopes.get(q)
            if s is not None and s.kind == "def":
                if head in s.params:
                    return "param"
                if head in s.locals:
                    return "local"
            q = q.rpartition(".")[0]
        if head in self._rawdefs(path):
            return "module"
        if head in self._bindings(path):
            return "import"
        return "builtin" if head in _BUILTINS else "unknown"

    def _resolve_chain(self, path: str, scope: str, dotted: str) -> _Res:
        head, *rest = dotted.split(".")
        kind = self._lookup(path, scope, head)
        if kind == "module":
            defs = self._rawdefs(path)
            for n in range(len(rest) + 1, 0, -1):
                d = defs.get(".".join([head, *rest][:n]))
                if d is not None:
                    return _Res("def", qual=d.qualified)
            return _UNKNOWN
        if kind == "import":
            return self._descend(self._import_res(path, self._bindings(path)[head]), rest, 0)
        return _UNKNOWN

    def _scope_at(self, path: str, lineno: int) -> str:
        best, best_line = MODULE_SCOPE, 0
        for s in self._scopes(path).values():
            if s.kind != "module" and s.lineno <= lineno <= s.end_lineno and s.lineno >= best_line:
                best, best_line = s.qualname, s.lineno
        return best

    # ---- public queries
    def resolve(self, path: str, name: str, lineno: int) -> Resolution:
        """What `name` means at `lineno`: locals and params shadow module and imported names."""
        p = _norm(path)
        try:
            scope = self._scope_at(p, lineno)
            kind = self._lookup(p, scope, name.split(".")[0])
            if kind not in ("module", "import"):
                return Resolution(kind)
            r = self._resolve_chain(p, scope, name)
            return Resolution(kind, self._target_of(r))
        except RecursionError:
            return Resolution("unknown")

    @staticmethod
    def _target_of(r: _Res) -> str | None:
        if r.status == "def":
            return r.qual
        if r.status == "module" and not r.target.startswith("dir:"):
            return r.target
        return None

    def import_target(self, path: str, module: str | None, level: int, name: str | None) -> str | None:
        """'path::name' that `from <level><module> import <name>` in `path` refers to, else None."""
        if not name:
            return None
        p = _norm(path)
        imp = Import("from", module or "", name, level, name, "", 0, False)
        r = self._import_res(p, imp)
        return r.qual if r.status == "def" else None

    def binding(self, path: str, scope: str, name: str) -> str | None:
        """Class qualname of a local/param/self in `scope` (one hop, cross-file), else None."""
        p = _norm(path)
        try:
            return self._binding(p, scope, name)
        except RecursionError:
            return None

    def _binding(self, p: str, scope: str, name: str) -> str | None:
        scopes = self._scopes(p)
        q = scope or MODULE_SCOPE
        while True:
            s = scopes.get(q)
            if s is not None and s.kind != "class":
                if s.cls and name == s.first_param:
                    return f"{p}::{s.cls}"
                if name in s.params:
                    return self._type_class(p, s.params[name])
                if name in s.locals:
                    src, text = s.locals[name]
                    if src == "ann":
                        return self._type_class(p, text)
                    if src == "call":
                        return self._call_type(p, q, text)
                    return None
            if q == MODULE_SCOPE:
                return None
            q = q.rpartition(".")[0] or MODULE_SCOPE

    def _class_qual(self, p: str, dotted: str) -> str | None:
        r = self._resolve_chain(p, MODULE_SCOPE, dotted)
        d = self._def(r.qual) if r.status == "def" else None
        return d.qualified if d is not None and d.kind == "class" else None

    def _type_class(self, p: str, text: str | None) -> str | None:
        head = _type_head(text)
        if head is None:
            return None
        r = self._resolve_chain(p, MODULE_SCOPE, head)
        d = self._def(r.qual) if r.status == "def" else None
        return d.qualified if d is not None and d.kind == "class" else None

    def _call_type(self, p: str, scope: str, callee: str) -> str | None:
        r = self._resolve_chain(p, scope, callee)
        d = self._def(r.qual) if r.status == "def" else None
        if d is None:
            return None
        if d.kind == "class":
            return d.qualified
        return self._type_class(d.path, d.returns) if d.kind == "def" else None

    def defs(self, path: str | None = None) -> tuple[Def, ...]:
        """Definitions (all files or one), sorted; `returns` resolved to a qualname when possible."""
        paths = self.files if path is None else (_norm(path),)
        out = []
        for p in paths:
            for d in self._rawdefs(p).values():
                got = self._type_class(p, d.returns) if d.returns else None
                bases = tuple(b for b in (self._class_qual(p, t) for t in d.bases) if b) if d.kind == "class" else ()
                out.append(replace(d, returns=got or d.returns, bases=bases))
        return tuple(sorted(out, key=lambda d: (d.path, d.lineno, d.qualname)))

    def _ref_parts(self, p: str, r: RawRef) -> tuple[str | None, str | None] | None:
        """(receiver, target) for a raw ref, or None when the ref is noise (builtin/unknown head)."""
        if r.kind == "key":
            head = (r.recv or "").split(".")[0]
            recv = self._binding(p, r.scope, head) if r.recv == head else None
            return (recv, None) if self._lookup(p, r.scope, head) in ("param", "local", "module", "import") else None
        head, _, _rest = r.name.partition(".")
        kind = self._lookup(p, r.scope, head)
        if kind in ("module", "import"):
            return None, self._target_of(self._resolve_chain(p, r.scope, r.name))
        if kind in ("param", "local") and _rest and r.kind in ("attr", "call"):
            recv_name, _, attr = r.name.rpartition(".")
            recv = self._binding(p, r.scope, recv_name) if "." not in recv_name else None
            if recv is None:
                return None, None
            d = self._def(f"{recv}.{attr}")
            return recv, d.qualified if d is not None else None
        return None

    def refs(self, path: str) -> tuple[Ref, ...]:
        """Resolved references in `path`: calls, attribute and name uses, key lookups, imports."""
        p = _norm(path)
        got = self._refs.get(p)
        if got is not None:
            return got
        f = self._facts(p)
        out: list[Ref] = []
        if f is not None:
            for r in f.refs:
                try:
                    parts = self._ref_parts(p, r)
                except RecursionError:
                    parts = None
                if parts is not None:
                    out.append(Ref(p, r.lineno, r.col, r.name, r.kind, parts[0], parts[1], r.scope))
            for imp in f.imports:
                if imp.kind == "star":
                    continue
                text = "." * imp.level + imp.module + ("." if imp.kind == "from" and imp.module else "") + (
                    imp.name if imp.kind == "from" else "")
                out.append(Ref(p, imp.line, 0, text, "import", None,
                               self._target_of(self._import_res(p, imp)), MODULE_SCOPE))
        got = tuple(sorted(out, key=lambda r: (r.lineno, r.col, r.kind, r.name)))
        self._refs[p] = got
        return got

    def users(self, qualname: str) -> tuple[Ref, ...]:
        """Every ref in the repo whose resolved target is `qualname` (provider -> consumers)."""
        if self._users is None:
            u: dict[str, list[Ref]] = {}
            for p in self.files:
                for r in self.refs(p):
                    if r.target:
                        u.setdefault(r.target, []).append(r)
            self._users = u
        return tuple(self._users.get(qualname, ()))

    # ---- extras
    def signature(self, qualname: str) -> Signature | None:
        d = self._def(qualname)
        return d.signature if d else None

    def call_sites(self, qualname: str) -> list[CallSite]:
        """Calls resolving to `qualname` with their argument shape, in (path, line) order."""
        if self._calls is None:
            cm: dict[str, list[CallSite]] = {}
            for p in self.files:
                f = self._facts(p)
                for r in (f.refs if f else ()):
                    if r.kind != "call":
                        continue
                    try:
                        parts = self._ref_parts(p, r)
                    except RecursionError:
                        continue
                    if parts and parts[1]:
                        cm.setdefault(parts[1], []).append(CallSite(p, r.lineno, r.nargs, r.kwnames, r.has_star))
            self._calls = {k: sorted(v, key=lambda c: (c.path, c.lineno, c.nargs, c.kwnames)) for k, v in cm.items()}
        return list(self._calls.get(qualname, ()))

    def dangling_refs(self, paths: Iterable[str | Path] | None = None) -> list[tuple[str, int, str, str]]:
        """(file, line, name, reason) for imports/uses of repo symbols that do not exist."""
        targets = self.files if paths is None else sorted({_norm(p) for p in paths})
        out: set[tuple[str, int, str, str]] = set()
        for p in targets:
            f = self._facts(p)
            if f is None:
                continue
            broken: set[str] = set()
            for imp in f.imports:
                if imp.optional or imp.kind == "star":
                    continue
                r = self._import_res(p, imp)
                if r.status == "missing":
                    text = "." * imp.level + imp.module + ("." if imp.kind == "from" and imp.module else "") + (
                        imp.name if imp.kind == "from" else "")
                    out.add((p, imp.line, text, r.reason))
                    broken.add(imp.local)
            for ref in f.refs:
                head = ref.name.split(".")[0]
                if ref.kind == "key" or head in broken or self._lookup(p, ref.scope, head) != "import":
                    continue
                try:
                    r = self._resolve_chain(p, ref.scope, ref.name)
                except RecursionError:
                    continue
                if r.status == "missing":
                    out.add((p, ref.lineno, ref.name, r.reason))
        return sorted(out)


__all__ = [
    "CROSSFILE_INDEX_ENV", "CallSite", "CrossFileIndex", "Def", "Import", "MAX_FILE_BYTES", "Param", "ParseError",
    "Ref", "Resolution", "Signature", "enabled", "extract_facts",
]
