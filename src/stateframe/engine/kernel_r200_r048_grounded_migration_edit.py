"""Advisory reuse of one observed Python call rewrite.

A partition has no write authority. The scalar edit receipts identify the
producer change and exemplar. Exact source revisions identify the proof inputs.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import inspect
import json
import re
import symtable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

HEADER = "[MIGRATION SITE PARTITION] "
EDIT_ACTION = "migration_site_scalar_edit"
DELIVERY_ACTION = "migration_site_partition_delivery"
MAX_SUMMARY_BYTES = 3_072
MAX_EDIT_BYTES = 2_048
MAX_SOURCE_BYTES = 256 * 1024
_DYNAMIC_SCOPE_NAMES = frozenset({"eval", "exec", "globals", "locals", "vars"})


def revision(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode()).hexdigest()


def _shape(node):
    return ast.dump(node) if node is not None else None


def _exemplar_comparison_tree(before, after):
    """Ignore a newly added postponed-annotations import when proving the call edit."""
    previous = [statement for statement in before.body
                if isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
                and any(alias.name == "annotations" for alias in statement.names)]
    additions = [statement for statement in after.body
                 if isinstance(statement, ast.ImportFrom)
                 and statement.module == "__future__"
                 and any(alias.name == "annotations" for alias in statement.names)]
    if (previous or len(additions) != 1
            or len(additions[0].names) != 1
            or additions[0].names[0].name != "annotations"
            or additions[0].names[0].asname is not None):
        return after
    comparison = copy.deepcopy(after)
    target = next((statement for statement in comparison.body
                   if _shape(statement) == _shape(additions[0])), None)
    if target is None:
        return after
    comparison.body.remove(target)
    return comparison


def _tree(source):
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class ScalarEdit:
    path: str
    before: str
    after: str
    offset: int
    old: str
    new: str
    receipt: str

    @classmethod
    def applied(cls, path, before, after, old, new, receipt):
        if not old or old == new or len((old + new).encode()) > MAX_EDIT_BYTES:
            return None
        offset = before.find(old)
        if (offset < 0 or before.find(old, offset + 1) >= 0
                or after != before[:offset] + new + before[offset + len(old):]):
            return None
        return cls(path, revision(before), revision(after), offset, old, new, receipt)

    def preimage(self, current):
        if current is None or revision(current) != self.after:
            return None
        end = self.offset + len(self.new)
        old_source = current[:self.offset] + self.old + current[end:]
        return old_source if revision(old_source) == self.before else None

    def evidence(self):
        return asdict(self)


@dataclass(frozen=True)
class ProducerDelta:
    path: str
    symbol: str
    kind: str
    old: str
    new: str
    # Defaults and parameter order are compared using static Python syntax.
    previous_default: str | None
    positional_index: int | None
    parameters: tuple[tuple[str, str, bool], ...]

    def binds(self, call):
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
                keyword.arg is None for keyword in call.keywords):
            return False
        keywords = {keyword.arg for keyword in call.keywords}
        if len(keywords) != len(call.keywords):
            return False
        parameters = [
            inspect.Parameter(name, getattr(inspect.Parameter, kind),
                              default=None if optional else inspect.Parameter.empty)
            for name, kind, optional in self.parameters
        ]
        try:
            inspect.Signature(parameters).bind(*call.args, **{key: None for key in keywords})
        except TypeError:
            return False
        return True

    def rewrite(self, call):
        if not self.binds(call):
            return None
        keywords = {keyword.arg for keyword in call.keywords}
        if self.positional_index is not None and len(call.args) > self.positional_index:
            return None
        result = copy.deepcopy(call)
        if self.kind == "keyword":
            if self.old not in keywords or self.new in keywords:
                return None
            next(keyword for keyword in result.keywords if keyword.arg == self.old).arg = self.new
        else:
            if self.old in keywords or self.previous_default is None:
                return None
            result.keywords.append(ast.keyword(
                arg=self.old, value=ast.parse(self.previous_default, mode="eval").body))
        return result


def _producer_functions(tree):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            if any(isinstance(part, ast.Return) and part.value is not None
                   for part in ast.walk(node)):
                yield node.name, node, False
        elif (isinstance(node, ast.ClassDef) and not node.bases and not node.keywords
              and not node.decorator_list):
            methods = [part for part in node.body if isinstance(part, ast.FunctionDef)
                       and part.name == "__init__"]
            if (len(methods) == 1 and not methods[0].decorator_list and not any(
                    isinstance(part, ast.FunctionDef) and part.name == "__new__"
                    or isinstance(part, ast.Name) and part.id in {"__init__", "__new__"}
                    and isinstance(part.ctx, (ast.Store, ast.Del))
                    for part in ast.walk(node))):
                yield node.name, methods[0], True


def _unique_producer(tree, symbol):
    definitions = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions += node.name == symbol
        elif (isinstance(node, ast.Name) and node.id == symbol
              and isinstance(node.ctx, (ast.Store, ast.Del))):
            return False
        elif isinstance(node, ast.alias) and (
                node.name == "*" or (node.asname or node.name.split(".")[0]) == symbol):
            return False
    return definitions == 1


def _parameters(fn):
    args = fn.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    return list(zip([*positional, *args.kwonlyargs], [*defaults, *args.kw_defaults], strict=True))


def _parameter_spec(fn, params, method):
    kinds = (["POSITIONAL_ONLY"] * len(fn.args.posonlyargs)
             + ["POSITIONAL_OR_KEYWORD"] * len(fn.args.args)
             + ["KEYWORD_ONLY"] * len(fn.args.kwonlyargs))
    return tuple((arg.arg, kind, default is not None)
                 for (arg, default), kind in list(zip(params, kinds, strict=True))[int(method):])


class _RenameLoads(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old, self.new = old, new

    def visit_Name(self, node):
        if node.id == self.old and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


def producer_delta(path: str, before: str, after: str) -> ProducerDelta | None:
    old_tree, new_tree = _tree(before), _tree(after)
    if old_tree is None or new_tree is None:
        return None
    for (symbol, old, method), (new_symbol, new, new_method) in zip(
            _producer_functions(old_tree), _producer_functions(new_tree)):
        if (symbol != new_symbol or method != new_method or _shape(old) == _shape(new)
                or not _unique_producer(old_tree, symbol)
                or old.args.vararg or old.args.kwarg or new.args.vararg or new.args.kwarg
                or len(old.args.posonlyargs) != len(new.args.posonlyargs)
                or len(old.args.args) != len(new.args.args)):
            continue
        old_params, new_params = _parameters(old), _parameters(new)
        if len(old_params) != len(new_params):
            continue
        different = [index for index, (left, right) in enumerate(zip(old_params, new_params))
                     if tuple(map(_shape, left)) != tuple(map(_shape, right))]
        if len(different) != 1:
            continue
        index = different[0]
        if index < max(int(method), len(old.args.posonlyargs)):
            continue
        (old_arg, old_default), (new_arg, new_default) = old_params[index], new_params[index]
        prior, updated = old_arg.arg, new_arg.arg
        # Restore the original callable in the resulting tree and require every
        # other statement to have survived. Then prove the callable's delta.
        saved = copy.deepcopy(new)
        new.args, new.body = old.args, old.body
        only_callable = _shape(old_tree) == _shape(new_tree)
        new.args, new.body = saved.args, saved.body
        if not only_callable or _shape(old_arg.annotation) != _shape(new_arg.annotation):
            continue
        positional = index - int(method) if index < len(old.args.posonlyargs) + len(old.args.args) else None
        parameters = _parameter_spec(old, old_params, method)
        if prior == updated:
            if (isinstance(old_default, ast.Constant)
                    and (new_default is None or isinstance(new_default, ast.Constant))
                    and tuple(map(_shape, old.body)) == tuple(map(_shape, new.body))):
                return ProducerDelta(path, symbol, "default", prior, updated,
                                     ast.unparse(old_default), positional, parameters)
            continue
        if (_shape(old_default) != _shape(new_default)
                or any(arg.arg == updated for arg, _ in old_params)
                or any(isinstance(node, (
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                    ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.ExceptHandler,
                    ast.Match, ast.Yield, ast.YieldFrom,
                )) or (isinstance(node, ast.Name) and (
                    node.id == updated or node.id in _DYNAMIC_SCOPE_NAMES
                    or (node.id == prior and not isinstance(node.ctx, ast.Load))))
                    for statement in old.body for node in ast.walk(statement))):
            continue
        equivalent = [_RenameLoads(prior, updated).visit(copy.deepcopy(row)) for row in old.body]
        if list(map(_shape, equivalent)) == list(map(_shape, new.body)):
            return ProducerDelta(path, symbol, "keyword", prior, updated, None, positional, parameters)
    return None


def producer_definition_matches(delta: ProducerDelta, observed: str, current: str) -> bool:
    """Require the grounded producer callable to remain unchanged at delivery."""
    observed_tree, current_tree = _tree(observed), _tree(current)
    if (observed_tree is None or current_tree is None
            or not _unique_producer(observed_tree, delta.symbol)
            or not _unique_producer(current_tree, delta.symbol)):
        return False

    def definition(tree):
        matches = [(node, method) for symbol, node, method in _producer_functions(tree)
                   if symbol == delta.symbol]
        return matches[0] if len(matches) == 1 else None

    observed_definition = definition(observed_tree)
    current_definition = definition(current_tree)
    if (observed_definition is None or current_definition is None
            or observed_definition[1] != current_definition[1]
            or _shape(observed_definition[0]) != _shape(current_definition[0])):
        return False
    observed_dependencies = _default_dependency_shapes(
        observed_tree, observed_definition[0])
    current_dependencies = _default_dependency_shapes(
        current_tree, current_definition[0])
    return (observed_dependencies is not None and current_dependencies is not None
            and observed_dependencies == current_dependencies)


def _default_dependency_shapes(tree, fn):
    """Bind names used by callable defaults to their module source expressions."""
    names = {
        part.id
        for _, default in _parameters(fn)
        if default is not None
        for part in ast.walk(default)
        if isinstance(part, ast.Name)
    }
    if not names:
        return ()

    bindings = {name: [] for name in names}

    def assigned_names(target):
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            return tuple(name for item in target.elts for name in assigned_names(item))
        return ()

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            bound = tuple(name for target in statement.targets
                          for name in assigned_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.NamedExpr)):
            bound = assigned_names(statement.target)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = (statement.name,)
        elif isinstance(statement, ast.Import):
            bound = tuple(alias.asname or alias.name.split(".")[0]
                          for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            bound = tuple(alias.asname or alias.name for alias in statement.names)
        else:
            bound = ()
        for name in names.intersection(bound):
            bindings[name].append(statement)

    builtin_names = vars(builtins)
    result = []
    for name in sorted(names):
        candidates = bindings[name]
        if len(candidates) == 1:
            shape = _shape(candidates[0])
        elif not candidates and name in builtin_names:
            shape = ("builtin", name)
        else:
            # Ambiguous or dynamic default dependencies cannot keep a decision alive.
            return None
        result.append((name, shape))
    return tuple(result)


def _module(path: str) -> str:
    return path.removeprefix("src/").removesuffix(".py").removesuffix("/__init__").replace("/", ".")


@dataclass(frozen=True)
class Caller:
    path: str
    revision: str
    line: int
    column: int
    end_line: int
    source: str
    context_source: str
    binding: str
    binding_source: str | None
    result_use: str
    forwarding: bool
    argument_shape: tuple[int, tuple[str | None, ...]]


class CallerFile:
    """Static binding for module imports and globals of ordinary functions."""

    def __init__(self, path: str, source: str):
        self.path, self.source = path, source
        self.revision = revision(source)
        self.tree = ast.parse(source)
        self.globals = symtable.symtable(source, path, "exec")
        self.nodes = tuple(ast.walk(self.tree))
        self.imports = tuple(node for node in self.nodes
                             if isinstance(node, (ast.Import, ast.ImportFrom)))
        self.parents = {child: node for node in self.nodes
                        for child in ast.iter_child_nodes(node)}

    def _scope_for(self, node):
        for scope in (self.globals, *self.globals.get_children()):
            if (scope is self.globals and node is self.tree) or (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and scope.get_lineno() == node.lineno and scope.get_name() == node.name):
                return scope
        return None

    def _binding(self, call, fn, delta):
        callee = call.func
        head = callee
        attributes = []
        while isinstance(head, ast.Attribute):
            attributes.insert(0, head.attr)
            head = head.value
        if not isinstance(head, ast.Name):
            return "unknown", None
        try:
            if fn is not None and not fn.lookup(head.id).is_global():
                return "unknown", None
            entry = self.globals.lookup(head.id)
        except KeyError:
            return "unknown", None
        if any((isinstance(node, ast.Global) and head.id in node.names)
               or (isinstance(node, ast.Name) and node.id in _DYNAMIC_SCOPE_NAMES)
               for node in self.nodes):
            return "unknown", None
        if attributes and any(
                isinstance(node, ast.Attribute) and node.attr == attributes[-1]
                and isinstance(node.ctx, (ast.Store, ast.Del))
                for node in self.nodes):
            return "unknown", None
        # Star imports or rebinding invalidate even an otherwise exact import.
        if any(isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
               for node in self.imports):
            return "unknown", None
        imported = []
        for node in self.imports:
            if isinstance(node, ast.ImportFrom):
                package = _module(self.path).split(".")
                if PurePosixPath(self.path).name != "__init__.py":
                    package.pop()
                if node.level and node.level > len(package):
                    continue
                base = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    base.append(node.module)
                for alias in node.names:
                    if (alias.asname or alias.name) == head.id:
                        imported.append((".".join([*base, alias.name, *attributes]), node))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    first, _, rest = alias.name.partition(".")
                    if (alias.asname or first) == head.id and (
                            alias.asname or not rest or attributes[:len(rest.split("."))] == rest.split(".")):
                        imported.append((".".join([alias.name if alias.asname else first, *attributes]), node))
        expected = f"{_module(delta.path)}.{delta.symbol}"
        if entry.is_imported() and not entry.is_assigned() and len(imported) == 1:
            qualified, node = imported[0]
            if qualified == expected and self.parents[node] is self.tree:
                return "producer", ast.get_source_segment(self.source, node)
            # A different import path can re-export the same object.
            return "unknown", ast.get_source_segment(self.source, node)
        definitions = [node for node in self.tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name == head.id]
        if (not attributes and len(definitions) == 1 and not entry.is_imported()
                and not any(isinstance(node, ast.Name) and node.id == head.id
                            and isinstance(node.ctx, (ast.Store, ast.Del))
                            for node in self.nodes)):
            node = definitions[0]
            if node.decorator_list:
                return "unknown", None
            identity = "producer" if self.path == delta.path and head.id == delta.symbol else "other"
            return identity, self.source.splitlines()[node.lineno - 1]
        return "unknown", None

    def relevant_calls(self, delta):
        names = {delta.symbol}
        names.update(alias.asname for node in self.imports if isinstance(node, ast.ImportFrom)
                     for alias in node.names if alias.name == delta.symbol and alias.asname)
        for node in self.nodes:
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if name in names or any(kw.arg == delta.old for kw in node.keywords):
                    yield node, self.facts(node, delta)

    def facts(self, call, delta):
        parent = self.parents[call]
        scope = parent
        while not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp,
                                     ast.DictComp, ast.GeneratorExp)):
            scope = self.parents[scope]
        fn = self._scope_for(scope)
        binding, binding_source = self._binding(call, fn if scope is not self.tree else None, delta)
        if fn is None:
            binding, binding_source = "unknown", None
        # Only Python's explicit expansion syntax proves argument forwarding.
        # A local name inside an ordinary expression (or positional argument)
        # does not make the call a forwarding site.
        forwarding = any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            keyword.arg is None for keyword in call.keywords)
        use = "expression"
        if isinstance(parent, ast.Expr):
            use = "discarded"
        elif isinstance(parent, ast.Return):
            use = "returned"
        elif isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            target = parent.targets[0]
            if isinstance(target, ast.Name):
                use = "assigned-used" if any(
                    isinstance(part, ast.Name) and part.id == target.id
                    and isinstance(part.ctx, ast.Load) for part in ast.walk(scope)
                ) else "assigned"
        context = parent
        while not isinstance(context, (ast.stmt, ast.Module)):
            context = self.parents[context]
        snippet = ast.get_source_segment(self.source, call) or ""
        context_source = ast.get_source_segment(self.source, context) or snippet
        return Caller(self.path, self.revision, call.lineno, call.col_offset,
                      call.end_lineno, snippet, context_source,
                      binding, binding_source, use, forwarding,
                      (len(call.args), tuple(keyword.arg for keyword in call.keywords)))


@dataclass(frozen=True)
class Decision:
    classification: str
    reason: str
    caller: Caller


def classify(delta: ProducerDelta, exemplar: Caller, call, caller: Caller) -> Decision:
    if caller.binding == "other":
        return Decision("unrelated", "different local binding", caller)
    if caller.binding != "producer":
        return Decision("unknown", "producer binding unproven", caller)
    if caller.forwarding:
        return Decision("exceptional", "forwarding changes argument evidence", caller)
    keywords = {keyword.arg for keyword in call.keywords}
    if (delta.kind == "keyword" and delta.new in keywords and delta.old not in keywords
            and (delta.positional_index is None
                 or len(call.args) <= delta.positional_index)):
        return Decision("unrelated", "caller already uses the renamed keyword", caller)
    if (delta.kind == "keyword" and delta.positional_index is not None
            and len(call.args) > delta.positional_index
            and delta.old not in keywords and delta.new not in keywords
            and delta.binds(call)):
        return Decision("unrelated", "renamed parameter is supplied positionally", caller)
    if (delta.kind == "keyword" and delta.old not in keywords
            and delta.new not in keywords and delta.binds(call)):
        return Decision("unrelated", "caller relies on the unchanged producer default", caller)
    if caller.result_use != exemplar.result_use or caller.result_use not in {
            "assigned", "returned", "discarded"}:
        return Decision("exceptional", "result use differs from exemplar", caller)
    if caller.argument_shape != exemplar.argument_shape or delta.rewrite(call) is None:
        return Decision("exceptional", "positional binding or defaults differ from exemplar", caller)
    return Decision("reusable", "same binding, rewrite and result use", caller)


def exemplar_for(delta, path, before, after):
    try:
        old_file, new_file = CallerFile(path, before), CallerFile(path, after)
    except (SyntaxError, ValueError, RecursionError):
        return None
    matches = []
    for call, site in old_file.relevant_calls(delta):
        if (site.binding != "producer" or site.forwarding
                or site.result_use not in {"assigned", "returned", "discarded"}):
            continue
        updated = delta.rewrite(call)
        if updated is None:
            continue
        args, keywords = call.args, call.keywords
        call.args, call.keywords = updated.args, updated.keywords
        transformed = _shape(old_file.tree) == _shape(
            _exemplar_comparison_tree(old_file.tree, new_file.tree))
        call.args, call.keywords = args, keywords
        if transformed:
            matches.append(site)
    return matches[0] if len(matches) == 1 else None


def _source_offset(source, line, column):
    lines = source.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        return None
    try:
        prefix = lines[line - 1].encode("utf-8")[:column].decode("utf-8")
    except UnicodeDecodeError:
        return None
    return sum(map(len, lines[:line - 1])) + len(prefix)


def _source_location(source, offset):
    if offset < 0 or offset > len(source):
        return None
    line = source.count("\n", 0, offset) + 1
    start = source.rfind("\n", 0, offset) + 1
    return line, len(source[start:offset].encode("utf-8"))


def _rewritten_exemplar_location(delta, exemplar, before, after, edit, after_file):
    try:
        before_file = CallerFile(exemplar.path, before)
    except (SyntaxError, ValueError, RecursionError):
        return None
    original = next((call for call, site in before_file.relevant_calls(delta)
                     if (site.line, site.column) == (exemplar.line, exemplar.column)), None)
    expected = delta.rewrite(original) if original is not None else None
    if expected is None:
        return None
    replacement_end = edit.offset + len(edit.new)
    matches = []
    for call, site in after_file.relevant_calls(delta):
        start = _source_offset(after, site.line, site.column)
        if (start is not None and start >= edit.offset
                and start + len(site.source) <= replacement_end
                and site.binding == exemplar.binding
                and site.binding_source == exemplar.binding_source
                and site.result_use == exemplar.result_use
                and _shape(call) == _shape(expected)):
            matches.append((start, site))
    return matches[0] if len(matches) == 1 else None


def current_exemplar_location(delta, exemplar, before, after, edit, following, current):
    """Locate the unchanged exemplar after receipt-backed edits in its file."""
    if (revision(before) != edit.before or revision(after) != edit.after
            or edit.preimage(after) != before):
        return None
    start = _source_offset(before, exemplar.line, exemplar.column)
    if start is None:
        return None
    mapped_anchor = None
    edit_end = edit.offset + len(edit.old)
    if edit_end <= start:
        start += len(edit.new) - len(edit.old)
    elif edit.offset <= start < edit_end:
        prefix_length = start - edit.offset
        if edit.old[:prefix_length] != edit.new[:prefix_length]:
            try:
                after_file = CallerFile(exemplar.path, after)
            except (SyntaxError, ValueError, RecursionError):
                return None
            mapped = _rewritten_exemplar_location(
                delta, exemplar, before, after, edit, after_file)
            if mapped is None:
                return None
            start, mapped_anchor = mapped
    location = _source_location(after, start)
    if location is None:
        return None
    try:
        after_file = CallerFile(exemplar.path, after)
    except (SyntaxError, ValueError, RecursionError):
        return None
    anchor = mapped_anchor or next((site for _, site in after_file.relevant_calls(delta)
                                    if (site.line, site.column) == location), None)
    if (anchor is None or anchor.binding != "producer"
            or anchor.binding_source != exemplar.binding_source
            or anchor.result_use != exemplar.result_use):
        return None
    end = start + len(anchor.source)

    source = after
    for later in following:
        if later.path != exemplar.path:
            continue
        later_end = later.offset + len(later.old)
        if (revision(source) != later.before
                or source[later.offset:later_end] != later.old):
            return None
        overlaps_anchor = later.offset < end and later_end > start
        if overlaps_anchor:
            # A scalar edit may use the exemplar as unchanged context while
            # narrowing its replacement to a neighboring caller. Re-anchor
            # only when that exact call survives uniquely on both sides.
            anchor_start = start - later.offset
            anchor_end = end - later.offset
            replacement_anchor = later.new.find(anchor.source)
            if (anchor_start < 0 or anchor_end > len(later.old)
                    or later.old[anchor_start:anchor_end] != anchor.source
                    or replacement_anchor < 0
                    or later.new.find(anchor.source, replacement_anchor + 1) >= 0):
                return None
            start = later.offset + replacement_anchor
            end = start + len(anchor.source)
        next_source = source[:later.offset] + later.new + source[later_end:]
        if revision(next_source) != later.after:
            return None
        if not overlaps_anchor and later_end <= start:
            shift = len(later.new) - len(later.old)
            start += shift
            end += shift
        source = next_source
    if source != current:
        return None
    location = _source_location(current, start)
    if location is None:
        return None
    try:
        current_file = CallerFile(exemplar.path, current)
    except (SyntaxError, ValueError, RecursionError):
        return None
    caller = next((site for _, site in current_file.relevant_calls(delta)
                   if (site.line, site.column) == location), None)
    if (caller is None or caller.source != anchor.source
            or caller.binding != anchor.binding
            or caller.binding_source != anchor.binding_source
            or caller.result_use != anchor.result_use):
        return None
    return location


@dataclass(frozen=True)
class Partition:
    item: str
    delta: ProducerDelta
    producer: ScalarEdit
    exemplar: ScalarEdit
    decisions: tuple[Decision, ...]

    def render(self):
        operation = asdict(self.delta)
        operation.pop("parameters")
        base = {
            "item": self.item, "producer": self.producer.evidence(),
            "callable": operation, "exemplar": self.exemplar.evidence(),
            "instruction": "Advisory only. Use the scalar edit tool on current source.",
        }
        base["partition"] = revision(_json([base, [
            (site.classification, site.caller.path, site.caller.revision,
             site.caller.line, site.caller.column) for site in self.decisions
        ]]))
        rows = []
        # Give an individually evidenced exception priority over reuse.
        order = {"exceptional": 0, "reusable": 1, "unknown": 2, "unrelated": 3}
        for decision in sorted(self.decisions, key=lambda row: order[row.classification]):
            row = asdict(decision)
            proposed = _json({**base, "sites": [*rows, row], "omitted": len(self.decisions)})
            if len((HEADER + proposed).encode()) <= MAX_SUMMARY_BYTES:
                rows.append(row)
        if not rows:
            return None
        return HEADER + _json({**base, "sites": rows, "omitted": len(self.decisions) - len(rows)})


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def named_migration(ticket: str, delta: ProducerDelta) -> bool:
    words = set(re.findall(r"\b\w+\b", ticket))
    return {delta.symbol, delta.old, delta.new}.issubset(words)


def source_edit_visible(edit: ScalarEdit, before: str, messages) -> bool:
    """Check that the preimage was actually delivered on the edit request."""
    from kernel_cache_layout_view import semantic_messages
    from kernel_read_exec import _visible_read_page
    from kernel_request_source_window import _range_is_covered, _verified_source_message_ranges

    start = before.count("\n", 0, edit.offset) + 1
    end = before.count("\n", 0, edit.offset + len(edit.old.rstrip("\n"))) + 1
    lines = before.splitlines(keepends=True)
    spans = []
    for row in semantic_messages(messages or ()):
        text = row.get("content")
        if not isinstance(text, str):
            continue
        if row.get("role") == "user":
            spans.extend(_verified_source_message_ranges(
                text, path=edit.path, source=before, revision=edit.before))
        if row.get("role") in {"tool", "user"} and (page := _visible_read_page(text)) is not None:
            first, last = page["start_line"], page["end_line"]
            if (page["path"] == edit.path and page["revision"] == edit.before
                    and page["start_column"] == 0):
                exact = "".join(lines[first - 1:last])
                if exact and last <= len(lines) and text.split("\n", 2)[-1].startswith(exact) and (
                        page["end_column"] is None or page["end_column"] == len(lines[last - 1])):
                    spans.append((first, last))
    return _range_is_covered(spans, start, end)


def parse_fact(line):
    if (not isinstance(line, str) or not line.startswith(HEADER)
            or len(line.encode()) > MAX_SUMMARY_BYTES):
        return None
    try:
        value = json.loads(line[len(HEADER):])
    except (ValueError, RecursionError):
        return None
    if (isinstance(value, dict) and isinstance(value.get("partition"), str)
            and isinstance(value.get("sites"), list) and all(
                isinstance(site, dict) and isinstance(site.get("classification"), str)
                and isinstance(site.get("caller"), dict) and all(
                    key in site["caller"]
                    for key in ("path", "revision", "line", "column", "end_line"))
                for site in value["sites"])):
        return value
    return None
