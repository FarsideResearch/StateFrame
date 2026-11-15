"""Non-executing Python interface and import analysis."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import sys
from typing import Protocol, Sequence

if __package__:
    from .kernel_parallel_records import (CanonicalSignature, ContractReport, ContractVerdict,
                                          ImportVerdict, InterfaceProjection, Param, UnitContract)
else:
    from kernel_parallel_records import (CanonicalSignature, ContractReport, ContractVerdict,
                                         ImportVerdict, InterfaceProjection, Param, UnitContract)


class Snapshot(Protocol):
    def read_bytes(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list_files(self) -> Sequence[str]: ...


_ALLOWED_DECORATORS = {'staticmethod', 'classmethod', 'property', 'functools.wraps', 'wraps',
                       'dataclass', 'dataclasses.dataclass'}
_OTHER_DEFINITIONS = {
    '.js': re.compile(r'^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)'),
    '.ts': re.compile(r'^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let|var)\s+(\w+)'),
    '.go': re.compile(r'^\s*(?:func|type|var|const)\s+(\w+)'),
    '.rs': re.compile(r'^\s*(?:pub\s+)?(?:fn|struct|enum|trait|const)\s+(\w+)'),
    '.java': re.compile(r'^\s*(?:public\s+)?(?:class|interface|enum|record)\s+(\w+)'),
}


def _other_definitions(data: bytes, path: str) -> list[tuple[str, int]] | None:
    pattern = _OTHER_DEFINITIONS.get(Path(path).suffix)
    if pattern is None or len(data) > (1 << 20):
        return None
    try:
        source = data.decode('utf-8')
    except UnicodeError:
        return None
    depth = 0
    block_comment = False
    found = []
    for number, line in enumerate(source.splitlines(), 1):
        start_depth = depth
        clean = []
        quote = None
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ''
            if block_comment:
                if char == '*' and next_char == '/':
                    block_comment = False
                    index += 2
                else:
                    index += 1
                clean.append(' ')
                continue
            if quote:
                if char == '\\':
                    index += 2
                    continue
                if char == quote:
                    quote = None
                index += 1
                clean.append(' ')
                continue
            if char == '/' and next_char == '*':
                block_comment = True
                clean.append(' ')
                index += 2
                continue
            if char == '/' and next_char == '/':
                break
            if char in ('"', "'", '`'):
                quote = char
                clean.append(' ')
                index += 1
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth < 0:
                    return None
            clean.append(char)
            index += 1
        if quote:
            return None
        match = pattern.match(''.join(clean))
        if match:
            if start_depth:
                return None
            found.append((match.group(1), number))
    return None if block_comment or depth else found


def _other_definition(data: bytes, path: str, symbol: str) -> int | None:
    definitions = _other_definitions(data, path)
    if definitions is None:
        return None
    return next((line for name, line in definitions if name == symbol), 0)


def _annotation(node: ast.expr | None) -> str | None:
    return ast.unparse(node).strip() if node is not None else None


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[str, ...] | None:
    values = []
    for decorator in node.decorator_list:
        base = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = ast.unparse(base)
        if name not in _ALLOWED_DECORATORS:
            return None
        values.append(name)
    return tuple(values)


def _parameters(args: ast.arguments) -> tuple[Param, ...]:
    result: list[Param] = []
    positional = [*args.posonlyargs, *args.args]
    first_default = len(positional) - len(args.defaults)
    for index, arg in enumerate(positional):
        kind = 'posonly' if index < len(args.posonlyargs) else 'pos'
        result.append(Param(arg.arg, kind, index >= first_default, _annotation(arg.annotation)))
    if args.vararg:
        result.append(Param(args.vararg.arg, 'varpos', False, _annotation(args.vararg.annotation)))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        result.append(Param(arg.arg, 'kwonly', default is not None, _annotation(arg.annotation)))
    if args.kwarg:
        result.append(Param(args.kwarg.arg, 'varkw', False, _annotation(args.kwarg.annotation)))
    return tuple(result)


class _Bindings(ast.NodeVisitor):
    """Count bindings in one scope without entering nested function or class bodies."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.count += 1

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.count += node.name == self.name
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.count += node.name == self.name
        for expression in (*node.bases, *node.keywords, *node.decorator_list):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # Comprehension loop targets have their own scope; assignment expressions do not.
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)

    def visit_Import(self, node: ast.Import) -> None:
        self.count += sum((alias.asname or alias.name.split('.')[0]) == self.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.count += sum(alias.name == '*' or (alias.asname or alias.name) == self.name
                          for alias in node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.count += node.name == self.name
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self.count += node.name == self.name
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self.count += node.name == self.name

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.count += node.rest == self.name
        self.generic_visit(node)


def _binding_count(body: list[ast.stmt], name: str) -> int:
    collector = _Bindings(name)
    for node in body:
        collector.visit(node)
    return collector.count


def _top_level(tree: ast.Module, name: str) -> ast.stmt | None:
    if _binding_count(tree.body, name) != 1:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node
        if isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == name for alias in node.names):
            return node
    return None


def _signature(tree: ast.Module, qualified: str) -> CanonicalSignature | str:
    parts = qualified.split('.')
    if not parts or any(not part.isidentifier() for part in parts) or len(parts) > 2:
        return 'UNKNOWN'
    node = _top_level(tree, parts[0])
    if node is None:
        return 'UNKNOWN'
    if len(parts) == 2:
        if not isinstance(node, ast.ClassDef):
            return 'UNKNOWN'
        if _binding_count(node.body, parts[1]) != 1:
            return 'UNKNOWN'
        matches = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == parts[1]]
        if len(matches) != 1:
            return 'UNKNOWN'
        node = matches[0]
        kind = 'method'
    else:
        kind = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        decorators = _decorators(node)
        if decorators is None:
            return 'UNKNOWN'
        return CanonicalSignature(kind or ('async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function'),
                                  _parameters(node.args), _annotation(node.returns), decorators)
    if isinstance(node, ast.ClassDef):
        decorators = _decorators(node)
        if decorators is None:
            return 'UNKNOWN'
        inits = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__']
        if (_binding_count(node.body, '__init__') != len(inits) or len(inits) > 1 or
                ('dataclass' in decorators or 'dataclasses.dataclass' in decorators) and not inits):
            return 'UNKNOWN'
        if inits and _decorators(inits[0]) is None:
            return 'UNKNOWN'
        return CanonicalSignature('class', _parameters(inits[0].args) if inits else (), None, decorators)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        if not isinstance(value, ast.Constant):
            return 'UNKNOWN'
        annotation = _annotation(node.annotation) if isinstance(node, ast.AnnAssign) else None
        return CanonicalSignature('constant', (), annotation, ())
    return 'UNKNOWN'


def canonical_signature(source: str, qualified: str) -> CanonicalSignature | str:
    try:
        return _signature(ast.parse(source), qualified)
    except (SyntaxError, ValueError, TypeError):
        return 'UNKNOWN'


def _module_map(snapshot: Snapshot) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(snapshot.list_files()):
        if not path.endswith('.py'):
            continue
        parts = path.split('/')
        if parts[0] == 'src':
            parts = parts[1:]
        if not parts:
            continue
        if parts[-1] == '__init__.py':
            name = '.'.join(parts[:-1])
        else:
            name = '.'.join([*parts[:-1], parts[-1][:-3]])
        if name:
            result.setdefault(name, path)
    return result


def _ambiguous_modules(snapshot: Snapshot) -> set[str]:
    found: dict[str, str] = {}
    ambiguous = set()
    for path in sorted(snapshot.list_files()):
        if not path.endswith('.py'):
            continue
        parts = path.split('/')
        if parts[0] == 'src':
            parts = parts[1:]
        name = '.'.join(parts[:-1] if parts[-1] == '__init__.py' else [*parts[:-1], parts[-1][:-3]])
        if name and name in found and found[name] != path:
            ambiguous.add(name)
        found[name] = path
    return ambiguous


def _absolute(module: str | None, level: int, path: str, mapping: dict[str, str]) -> str | None:
    if not level:
        return module or ''
    current = next((name for name, file in mapping.items() if file == path), None)
    if current is None:
        return None
    package = current if path.endswith('/__init__.py') else current.rpartition('.')[0]
    parts = package.split('.') if package else []
    if level > len(parts) + 1:
        return None
    base = parts[:len(parts) - level + 1]
    return '.'.join([*base, *((module or '').split('.') if module else [])])


def _module_status(name: str | None, mapping: dict[str, str]) -> str:
    if name is None or not name:
        return 'UNRESOLVED'
    if name in mapping or any(key.startswith(name + '.') for key in mapping):
        return 'RESOLVED'
    if name.split('.')[0] in sys.stdlib_module_names:
        return 'EXTERNAL'
    return 'EXTERNAL' if name.split('.')[0] not in {key.split('.')[0] for key in mapping} else 'UNRESOLVED'


def resolve_imports(snapshot: Snapshot, paths: Sequence[str]) -> tuple[ImportVerdict, ...]:
    mapping = _module_map(snapshot)
    ambiguous = _ambiguous_modules(snapshot)
    verdicts = []
    for path in sorted(paths):
        if not path.endswith('.py'):
            continue
        try:
            tree = ast.parse(snapshot.read_bytes(path).decode('utf-8'))
        except (UnicodeError, SyntaxError, OSError):
            verdicts.append(ImportVerdict(path, 0, '', 'UNKNOWN'))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    verdicts.append(ImportVerdict(path, node.lineno, alias.name,
                                                  'UNKNOWN' if alias.name in ambiguous else _module_status(alias.name, mapping)))
            elif isinstance(node, ast.ImportFrom):
                module = _absolute(node.module, node.level, path, mapping)
                status = _module_status(module, mapping)
                for alias in node.names:
                    candidate = f'{module}.{alias.name}' if module else alias.name
                    if alias.name == '*':
                        final = 'UNKNOWN' if status == 'RESOLVED' else status
                    elif module in ambiguous or candidate in ambiguous:
                        final = 'UNKNOWN'
                    elif candidate in mapping:
                        final = 'RESOLVED'
                    elif status == 'RESOLVED' and module in mapping:
                        try:
                            source = snapshot.read_bytes(mapping[module]).decode('utf-8')
                            tree = ast.parse(source)
                            resolved, _, _, _ = _resolve_symbol(snapshot, mapping[module], alias.name, set())
                            if resolved != 'UNKNOWN':
                                final = 'RESOLVED'
                            elif _top_level(tree, alias.name) is not None or any(
                                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == '__getattr__'
                                    or isinstance(item, ast.ImportFrom) and any(a.name == '*' for a in item.names)
                                    for item in tree.body):
                                final = 'UNKNOWN'
                            else:
                                final = 'UNRESOLVED'
                        except (OSError, UnicodeError, SyntaxError):
                            final = 'UNKNOWN'
                    else:
                        final = 'UNRESOLVED' if status == 'RESOLVED' else status
                    verdicts.append(ImportVerdict(path, node.lineno, candidate, final))
    return tuple(sorted(verdicts, key=lambda item: (item.path, item.line, item.module)))


def _resolve_symbol(snapshot: Snapshot, path: str, symbol: str, seen: set[tuple[str, str]]) -> tuple[CanonicalSignature | str, str, int, str]:
    if (path, symbol) in seen:
        return 'UNKNOWN', path, 0, ''
    seen.add((path, symbol))
    source = snapshot.read_bytes(path).decode('utf-8')
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 'UNKNOWN', path, 0, ''
    node = _top_level(tree, symbol.split('.')[0])
    signature = _signature(tree, symbol)
    all_assignments = [item for item in tree.body if isinstance(item, (ast.Assign, ast.AnnAssign))
                       and any(isinstance(target, ast.Name) and target.id == '__all__'
                               for target in (item.targets if isinstance(item, ast.Assign) else [item.target]))]
    if all_assignments:
        if len(all_assignments) != 1:
            return 'UNKNOWN', path, 0, ''
        all_value = all_assignments[0].value
        if not isinstance(all_value, (ast.List, ast.Tuple)) or any(
                not isinstance(element, ast.Constant) or not isinstance(element.value, str)
                for element in all_value.elts):
            return 'UNKNOWN', path, 0, ''
        if symbol.split('.')[0] not in {element.value for element in all_value.elts}:
            return 'UNKNOWN', path, 0, ''
    if isinstance(node, ast.ImportFrom) and len(symbol.split('.')) == 1:
        names = [alias for alias in node.names if (alias.asname or alias.name) == symbol]
        if len(names) != 1 or names[0].name == '*':
            return 'UNKNOWN', path, node.lineno, ''
        mapping = _module_map(snapshot)
        module = _absolute(node.module, node.level, path, mapping)
        target = mapping.get(module or '')
        if target is None:
            return 'UNKNOWN', path, node.lineno, ''
        return _resolve_symbol(snapshot, target, names[0].name, seen)
    if node is None:
        return 'UNKNOWN', path, 0, ''
    doc = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
    return signature, path, node.lineno, (doc or '').splitlines()[0] if doc else ''


def project_interface(snapshot: Snapshot, path: str, symbol: str, export_id: str | None) -> InterfaceProjection:
    data = snapshot.read_bytes(path)
    if path.endswith('.py'):
        signature, defining_path, line, doc = _resolve_symbol(snapshot, path, symbol, set())
        signature = None if signature == 'UNKNOWN' else signature
        data = snapshot.read_bytes(defining_path)
        return InterfaceProjection(export_id, defining_path, symbol, signature, doc, line, hashlib.sha256(data).hexdigest())
    return InterfaceProjection(export_id, path, symbol, None, '',
                               _other_definition(data, path, symbol) or 0, hashlib.sha256(data).hexdigest())


def _signature_reason(expected: CanonicalSignature, actual: CanonicalSignature) -> str:
    if (expected.kind != actual.kind or expected.decorators != actual.decorators or
            len(expected.params) != len(actual.params)):
        return 'SIGNATURE_MISMATCH'
    unverified = bool(expected.returns is not None and actual.returns is None)
    if (expected.returns is not None and actual.returns is not None and
            expected.returns != actual.returns):
        return 'SIGNATURE_MISMATCH'
    for declared, found in zip(expected.params, actual.params):
        if ((declared.name, declared.kind, declared.has_default) !=
                (found.name, found.kind, found.has_default)):
            return 'SIGNATURE_MISMATCH'
        if declared.annotation is not None and found.annotation is None:
            unverified = True
        elif (declared.annotation is not None and found.annotation is not None and
              declared.annotation != found.annotation):
            return 'SIGNATURE_MISMATCH'
    return 'annotation_unverified' if unverified else 'MATCH'


def verify_contracts(contracts: Sequence[UnitContract], snapshot: Snapshot, *, unit_id: str) -> ContractReport:
    unit = next((item for item in contracts if item.id == unit_id), None)
    if unit is None:
        raise ValueError(f'unknown subframe {unit_id!r}')
    verdicts = []
    for export in unit.exports:
        if not snapshot.exists(export.path):
            verdicts.append(ContractVerdict(export.id, 'FAIL', export.signature, None, 'MISSING_EXPORT'))
            continue
        if not export.path.endswith('.py'):
            data = snapshot.read_bytes(export.path)
            location = _other_definition(data, export.path, export.symbol)
            reason = ('DEFINITION_PRESENT' if location else 'DEFINITION_MISSING') if location is not None else 'UNSUPPORTED_LANGUAGE_OR_SIZE'
            verdicts.append(ContractVerdict(export.id, 'UNKNOWN', export.signature, None, reason))
            continue
        try:
            actual, _, line, _ = _resolve_symbol(snapshot, export.path, export.symbol, set())
        except (UnicodeError, OSError):
            actual, line = 'UNKNOWN', 0
        if actual == 'UNKNOWN':
            try:
                tree = ast.parse(snapshot.read_bytes(export.path).decode('utf-8'))
                target_name = export.symbol.split('.')[0]
                present = any(
                    (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and node.name == target_name) or
                    (isinstance(node, ast.ImportFrom) and any(
                        (alias.asname or alias.name) == target_name for alias in node.names)) or
                    (isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                        isinstance(target, ast.Name) and target.id == target_name
                        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])))
                    for node in ast.walk(tree))
            except (UnicodeError, SyntaxError, OSError):
                present = True
            verdicts.append(ContractVerdict(export.id, 'UNKNOWN' if line or present else 'FAIL', export.signature,
                                            None, 'UNSUPPORTED_STATIC_FORM' if line or present else 'MISSING_EXPORT'))
        elif export.signature is None:
            verdicts.append(ContractVerdict(export.id, 'UNKNOWN', None, actual, 'SIGNATURE_UNDECLARED'))
        else:
            reason = _signature_reason(export.signature, actual)
            verdicts.append(ContractVerdict(export.id, 'FAIL' if reason == 'SIGNATURE_MISMATCH' else 'PASS',
                                            export.signature, actual, reason))
    imports = resolve_imports(snapshot, [*unit.owns, *unit.creates])
    return ContractReport(tuple(verdicts), imports,
                          all(item.status != 'FAIL' for item in verdicts) and
                          all(item.status != 'UNRESOLVED' for item in imports))
