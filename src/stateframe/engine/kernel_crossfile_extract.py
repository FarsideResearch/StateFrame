"""Single-file fact extraction for the cross-file index (pure: source text in, facts out).

Records defs (with signatures, return annotations, fields), imports, scopes (params and
locals with their binding source) and raw refs.  Cross-file resolution lives in
kernel_crossfile_index; nothing here touches the filesystem.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable

MODULE_SCOPE = "<module>"
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: str  # posonly | pos | kwonly
    has_default: bool


@dataclass(frozen=True, slots=True)
class Signature:
    params: tuple[Param, ...]
    varargs: str | None = None
    varkw: str | None = None


@dataclass(frozen=True, slots=True)
class Def:
    path: str
    qualname: str
    kind: str  # def | class | const | var
    lineno: int
    end_lineno: int
    signature: Signature | None = None
    returns: str | None = None
    fields: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()  # classes: resolved base qualnames (raw base text until indexed)

    @property
    def qualified(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass(frozen=True, slots=True)
class Import:
    kind: str  # import | from | star
    module: str
    name: str
    level: int
    local: str
    bound_module: str
    line: int
    optional: bool


@dataclass(frozen=True, slots=True)
class RawRef:
    lineno: int
    col: int
    name: str            # dotted text as written (the key text for kind=key)
    kind: str            # call | attr | name | key
    scope: str
    recv: str | None = None   # key refs: receiver chain text
    nargs: int = 0
    kwnames: tuple[str, ...] = ()
    has_star: bool = False


@dataclass(frozen=True, slots=True)
class Scope:
    qualname: str
    kind: str            # module | def | class
    lineno: int
    end_lineno: int
    params: dict[str, str | None] = field(default_factory=dict)   # name -> annotation text
    locals: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (ann|call|local, text)
    cls: str | None = None          # enclosing class qualname for a method
    first_param: str | None = None  # self/cls name of a method


@dataclass(frozen=True, slots=True)
class FileFacts:
    defs: tuple[Def, ...]
    imports: tuple[Import, ...]
    scopes: tuple[Scope, ...]
    refs: tuple[RawRef, ...]


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef, drop_first: bool = False) -> Signature:
    a = fn.args
    pos = [*a.posonlyargs, *a.args]
    nd = len(a.defaults)
    params = [Param(x.arg, "posonly" if i < len(a.posonlyargs) else "pos", i >= len(pos) - nd)
              for i, x in enumerate(pos)]
    params += [Param(x.arg, "kwonly", d is not None) for x, d in zip(a.kwonlyargs, a.kw_defaults)]
    if drop_first:
        for i, p in enumerate(params):
            if p.kind != "kwonly":
                del params[i]
                break
    return Signature(tuple(params), a.vararg.arg if a.vararg else None, a.kwarg.arg if a.kwarg else None)


def _flat(stmts: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Statements at this level, descending into if/try/with but not defs or classes."""
    for s in stmts:
        if isinstance(s, ast.If):
            yield from _flat(s.body)
            yield from _flat(s.orelse)
        elif isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            yield from _flat(s.body)
            for h in s.handlers:
                yield from _flat(h.body)
            yield from _flat(s.orelse)
            yield from _flat(s.finalbody)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            yield from _flat(s.body)
        else:
            yield s


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in _target_names(e)]
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return []


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (ValueError, RecursionError):
        return None


def _var_kind(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    return "const" if leaf.isupper() and any(c.isalpha() for c in leaf) else "var"


def _self_fields(cls: ast.ClassDef) -> set[str]:
    """Attributes assigned as <first-param>.x inside the class's methods."""
    out: set[str] = set()
    for m in _flat(cls.body):
        if not isinstance(m, _FUNCS) or not (m.args.posonlyargs or m.args.args):
            continue
        if any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in m.decorator_list):
            continue
        first = (m.args.posonlyargs or m.args.args)[0].arg
        for n in ast.walk(m):
            if (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                    and isinstance(n.value, ast.Name) and n.value.id == first):
                out.add(n.attr)
    return out


def _collect_defs(stmts: Iterable[ast.stmt], prefix: str, in_class: bool, out: list[Def]) -> None:
    for s in _flat(stmts):
        end = getattr(s, "end_lineno", None) or s.lineno
        if isinstance(s, _FUNCS):
            out.append(Def("", prefix + s.name, "def", s.lineno, end, _signature(s), _unparse(s.returns)))
        elif isinstance(s, ast.ClassDef):
            init = next((b for b in _flat(s.body) if isinstance(b, _FUNCS) and b.name == "__init__"), None)
            fields = {n for b in _flat(s.body) for n in _stmt_names(b)} | _self_fields(s)
            out.append(Def("", prefix + s.name, "class", s.lineno, end,
                           _signature(init, drop_first=True) if init is not None else None,
                           None, tuple(sorted(fields)),
                           tuple(t for t in (_unparse(b) for b in s.bases) if t)))
            _collect_defs(s.body, f"{prefix}{s.name}.", True, out)
        else:
            out.extend(Def("", prefix + n, _var_kind(n), s.lineno, end) for n in _stmt_names(s))


def _stmt_names(s: ast.stmt) -> list[str]:
    if isinstance(s, ast.Assign):
        return [n for t in s.targets for n in _target_names(t)]
    if isinstance(s, ast.AnnAssign):
        return _target_names(s.target)
    return []


# ---- imports
def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")
               for n in nodes)


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[Import] = []
        self._optional = 0

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_catches_import_error(h) for h in node.handlers)
        self._optional += guarded
        for s in node.body:
            self.visit(s)
        self._optional -= guarded
        for part in (node.handlers, node.orelse, node.finalbody):
            for s in part:
                self.visit(s)

    visit_TryStar = visit_Try

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            top = a.name.split(".")[0]
            self.imports.append(Import("import", a.name, "", 0, a.asname or top,
                                       a.name if a.asname else top, node.lineno, bool(self._optional)))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            self.imports.append(Import("star" if a.name == "*" else "from", node.module or "", a.name, node.level,
                                       a.asname or a.name, "", node.lineno, bool(self._optional)))


# ---- scopes and refs
def chain(node: ast.AST) -> tuple[str, str] | None:
    """('a.b.c', 'a') for a pure attribute chain rooted at a Name, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts)), node.id
    return None


def _walk_scope(stmts: Iterable[ast.AST]) -> Iterable[ast.AST]:
    """Nodes of one scope in source order; nested defs yield themselves but are not entered."""
    for n in stmts:
        yield n
        if not isinstance(n, (*_DEFS, ast.Lambda)):
            yield from _walk_scope(ast.iter_child_nodes(n))


def _value_src(value: ast.AST | None) -> tuple[str, str]:
    if isinstance(value, ast.Await):
        value = value.value
    if isinstance(value, ast.Call):
        c = chain(value.func)
        if c is not None:
            return ("call", c[0])
    return ("local", "")


def _locals_of(body: Iterable[ast.AST]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}

    def put(name: str, src: tuple[str, str]) -> None:
        cur = out.get(name)
        if cur is None or (cur[0] == "local" and src[0] != "local") or src[0] == "ann":
            out[name] = src

    for n in _walk_scope(body):
        if isinstance(n, ast.Assign):
            single = len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            for t in n.targets:
                for nm in _target_names(t):
                    put(nm, _value_src(n.value) if single else ("local", ""))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            ann = _unparse(n.annotation)
            put(n.target.id, ("ann", ann) if ann else ("local", ""))
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                if isinstance(it.optional_vars, ast.Name):
                    put(it.optional_vars.id, _value_src(it.context_expr))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            put(n.id, ("local", ""))
        elif isinstance(n, ast.ExceptHandler) and n.name:
            put(n.name, ("local", ""))
        elif isinstance(n, _DEFS):
            put(n.name, ("local", ""))
    return out


def _lambda_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    args = getattr(node, "args", None)
    if isinstance(args, ast.arguments):
        for x in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if x is not None:
                names.add(x.arg)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
    return names


class _Extractor(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.refs: list[RawRef] = []
        self.scopes: list[Scope] = [Scope(MODULE_SCOPE, "module", 1, 10 ** 9, {}, _locals_of(tree.body))]
        self._stack: list[tuple[str, str]] = [(MODULE_SCOPE, "module")]
        self._shadow: list[set[str]] = []

    @property
    def _scope(self) -> str:
        return self._stack[-1][0]

    def _live(self, root: str) -> bool:
        return not any(root in s for s in self._shadow)

    def _push(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        parent, pkind = self._stack[-1]
        qual = node.name if parent == MODULE_SCOPE else f"{parent}.{node.name}"
        end = getattr(node, "end_lineno", None) or node.lineno
        if isinstance(node, ast.ClassDef):
            sc = Scope(qual, "class", node.lineno, end)
            kind = "class"
        else:
            a = node.args
            pos = [*a.posonlyargs, *a.args]
            params = {x.arg: _unparse(x.annotation) for x in [*pos, *a.kwonlyargs]}
            for x in (a.vararg, a.kwarg):
                if x is not None:
                    params[x.arg] = None
            static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list)
            method = pkind == "class" and not static and bool(pos)
            sc = Scope(qual, "def", node.lineno, end, params, _locals_of(node.body),
                       parent if method else None, pos[0].arg if method else None)
            kind = "def"
        self.scopes.append(sc)
        self._stack.append((qual, kind))

    def _def(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        if isinstance(node, ast.ClassDef):
            for b in [*node.bases, *(k.value for k in node.keywords)]:
                self.visit(b)
        else:
            a = node.args
            for x in [*a.defaults, *(d for d in a.kw_defaults if d is not None), node.returns,
                      *(p.annotation for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
                        if p is not None)]:
                if x is not None:
                    self.visit(x)
        self._push(node)
        for s in node.body:
            self.visit(s)
        self._stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def

    def _shadowed(self, node: ast.AST) -> None:
        self._shadow.append(_lambda_names(node))
        self.generic_visit(node)
        self._shadow.pop()

    visit_Lambda = visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _shadowed

    def _add(self, node: ast.AST, name: str, kind: str, **kw: object) -> None:
        self.refs.append(RawRef(node.lineno, node.col_offset, name, kind, self._scope, **kw))  # type: ignore[arg-type]

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and self._live(node.id):
            self._add(node, node.id, "name")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        c = chain(node)
        if c is None:
            self.generic_visit(node)
        elif isinstance(node.ctx, ast.Load) and self._live(c[1]):
            self._add(node, c[0], "attr")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        c = chain(node.value)
        key = node.slice
        if c is not None and isinstance(key, ast.Constant) and isinstance(key.value, str) and self._live(c[1]):
            self._add(node, key.value, "key", recv=c[0])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        c = chain(node.func)
        if c is None or not self._live(c[1]):
            self.generic_visit(node)
            return
        star = any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords)
        self._add(node, c[0], "call", nargs=sum(not isinstance(a, ast.Starred) for a in node.args),
                  kwnames=tuple(k.arg for k in node.keywords if k.arg is not None), has_star=star)
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            rc = chain(f.value)
            if rc is not None:
                self._add(node, node.args[0].value, "key", recv=rc[0])
        for a in node.args:
            self.visit(a)
        for k in node.keywords:
            self.visit(k.value)


def extract_facts(source: str, *, defs_only: bool = False) -> FileFacts:
    """Parse one file's source (raises SyntaxError/ValueError/RecursionError on bad input)."""
    tree = ast.parse(source)
    defs: list[Def] = []
    _collect_defs(tree.body, "", False, defs)
    iv = _ImportVisitor()
    iv.visit(tree)
    if defs_only:
        return FileFacts(tuple(defs), tuple(iv.imports), (), ())
    ex = _Extractor(tree)
    ex.visit(tree)
    return FileFacts(tuple(defs), tuple(iv.imports), tuple(ex.scopes), tuple(ex.refs))
