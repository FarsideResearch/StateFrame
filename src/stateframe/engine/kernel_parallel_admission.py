"""Pure admission decisions over a file-backed or supplied snapshot."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import heapq
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Callable, Iterable, Protocol, Sequence

if __package__:
    from .kernel_parallel_contracts import _ambiguous_modules, _module_map, _other_definitions, _resolve_symbol, project_interface
    from .kernel_parallel_records import (AdmissionAdjustment, AdmissionResult, ConsumeSpec,
                                          CouplingEdge, CouplingPair, Evidence,
                                          ParallelLimits, ParallelRequest,
                                          Refusal, RequirementClause, UnitContract, canonical_json,
                                          format_parallel_limit)
else:
    from kernel_parallel_contracts import _ambiguous_modules, _module_map, _other_definitions, _resolve_symbol, project_interface
    from kernel_parallel_records import (AdmissionAdjustment, AdmissionResult, ConsumeSpec,
                                         CouplingEdge, CouplingPair, Evidence,
                                          ParallelLimits, ParallelRequest,
                                          Refusal, RequirementClause, UnitContract, canonical_json,
                                          format_parallel_limit)


class Snapshot(Protocol):
    root: Path
    generation: int
    manifest_digest: str
    def read_bytes(self, path: str) -> bytes: ...
    def exists(self, path: str) -> bool: ...
    def list_files(self) -> Iterable[str]: ...


@dataclass(frozen=True)
class RescanResult:
    refusal: Refusal | None
    pairs: tuple[CouplingPair, ...]


class DirectorySnapshot:
    def __init__(self, root: Path, generation: int):
        self.root = Path(root)
        self.generation = generation
        digest = hashlib.sha256()
        for name in self.list_files():
            path = self.root / name
            mode = path.lstat().st_mode
            if stat.S_ISREG(mode):
                kind = 'file'
                hasher = hashlib.sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda: source.read(1 << 20), b''):
                        hasher.update(chunk)
                sha256 = hasher.hexdigest()
            elif stat.S_ISLNK(mode):
                kind = 'symlink'
                sha256 = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            else:
                kind = 'special'
                sha256 = ''
            digest.update(canonical_json({'path': name, 'kind': kind,
                                          'mode': stat.S_IMODE(mode), 'sha256': sha256}))
            digest.update(b'\n')
        self.manifest_digest = digest.hexdigest()

    def read_bytes(self, path: str) -> bytes:
        return (self.root / path).read_bytes()

    def exists(self, path: str) -> bool:
        return (self.root / path).exists()

    def list_files(self) -> Iterable[str]:
        return sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*')
                      if path.is_symlink() or not path.is_dir())


_FORBIDDEN = {'.git', '.hg', '.svn', '.venv', 'venv', 'env', '__pycache__', '.pytest_cache',
              '.mypy_cache', '.ruff_cache', '.tox', 'node_modules'}
def _refuse(snapshot: Snapshot, limits: ParallelLimits, reason: str, detail: str = '',
            paths: Iterable[str] = (), evidence: Iterable[Evidence] = (), members: Iterable[str] = (),
            adjustments: Iterable[AdmissionAdjustment] = ()) -> AdmissionResult:
    return AdmissionResult('REFUSED', Refusal(reason, detail, tuple(sorted(paths)), tuple(evidence),
                                               tuple(sorted(members))), (), (), (),
                           snapshot.generation, snapshot.manifest_digest, limits,
                           adjustments=tuple(adjustments))


def _limit_detail(name: str, value: int) -> str:
    return format_parallel_limit(name, value)


def _path_overlap(left: str, right: str, *, case_insensitive: bool) -> bool:
    a = left.casefold() if case_insensitive else left
    b = right.casefold() if case_insensitive else right
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def _path_safety_reason(path: str, root: Path, *, create: bool) -> tuple[str | None, tuple[str, ...]]:
    if not isinstance(path, str) or not path or '\\' in path or '\0' in path or path.startswith('/'):
        return 'path must be a normalized workspace-relative POSIX path', ()
    parts = path.split('/')
    if any(part in ('', '.', '..') for part in parts):
        return 'path contains an empty, current, or parent component', ()
    if any(part in _FORBIDDEN for part in parts):
        return 'path contains a forbidden repository directory', ()
    if PurePosixPath(path).as_posix() != path:
        return 'path is not normalized', ()
    current = root
    missing_ancestors = []
    for index, part in enumerate(parts):
        current = current / part
        try:
            file_stat = current.lstat()
        except FileNotFoundError:
            if index < len(parts) - 1:
                ancestor = '/'.join(parts[:index + 1])
                if not create:
                    return f'missing ancestor directory {ancestor}', tuple(missing_ancestors)
                missing_ancestors.append(ancestor)
            continue
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            component = '/'.join(parts[:index + 1])
            return f'cannot inspect path component {component}: {reason}', tuple(missing_ancestors)
        mode = file_stat.st_mode
        if stat.S_ISLNK(mode):
            component = '/'.join(parts[:index + 1])
            return f'path component {component} is a symlink', tuple(missing_ancestors)
        component = '/'.join(parts[:index + 1])
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            kind = 'regular file' if stat.S_ISREG(mode) else 'special file'
            return f'ancestor {component} is a {kind}, not a directory', tuple(missing_ancestors)
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            kind = 'directory' if stat.S_ISDIR(mode) else 'special file'
            return f'target {component} is a {kind}, not a regular file', tuple(missing_ancestors)
        if index == len(parts) - 1 and not create and file_stat.st_nlink > 1:
            return f'owned file {component} has hardlink aliases', tuple(missing_ancestors)
    return None, tuple(missing_ancestors)


def _safe_path(path: str, root: Path, *, create: bool) -> bool:
    reason, _ = _path_safety_reason(path, root, create=create)
    return reason is None


def _definitions(path: str, data: bytes) -> set[str] | None:
    if path.endswith('.py'):
        try:
            tree = ast.parse(data.decode('utf-8'))
        except (UnicodeError, SyntaxError):
            return None
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(target.id for target in targets if isinstance(target, ast.Name))
        return names
    definitions = _other_definitions(data, path)
    return None if definitions is None else {name for name, _ in definitions}


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _scope_bindings(scope: ast.AST) -> tuple[set[str], set[str]]:
    """Names bound directly in `scope` (not in nested scopes) and its global/nonlocal declarations."""
    bound: set[str] = set()
    declared: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg is not None:
                bound.add(arg.arg)
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.update(param.name for param in getattr(scope, 'type_params', ()))
    is_comprehension = isinstance(scope, _COMPREHENSIONS)

    def walk(node: ast.AST, top: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(child.name)
                if isinstance(child, _COMPREHENSIONS):
                    # A walrus inside a nested comprehension binds in the nearest non-comprehension scope.
                    if not is_comprehension:
                        bound.update(n.target.id for n in ast.walk(child)
                                     if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name))
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                bound.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                bound.update(alias.asname or alias.name.split('.')[0] for alias in child.names)
            elif isinstance(child, (ast.MatchAs, ast.MatchStar)) and child.name:
                bound.add(child.name)
            elif isinstance(child, ast.MatchMapping) and child.rest:
                bound.add(child.rest)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                declared.update(child.names)
            walk(child, False)

    if is_comprehension:
        for generator in scope.generators:  # type: ignore[attr-defined]
            for node in ast.walk(generator.target):
                if isinstance(node, ast.Name):
                    bound.add(node.id)
    else:
        walk(scope, True)
    return bound, declared


def _module_bindings(tree: ast.Module) -> set[str]:
    bound, _ = _scope_bindings(tree)
    # `global x` + assignment inside a function binds a module name too.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local, declared = _scope_bindings(node)
            bound.update(declared & local)
    return bound


def _free_global_loads(tree: ast.Module) -> list[ast.Name]:
    """Name loads that resolve to a module global the consumer module does not bind itself."""
    module_bound = _module_bindings(tree)
    found: list[ast.Name] = []

    def resolves_local(name: str, stack: list[tuple[ast.AST, set[str], set[str]]]) -> bool:
        for index in range(len(stack) - 1, -1, -1):
            scope, bound, declared = stack[index]
            if name in declared:
                return False
            if isinstance(scope, ast.ClassDef) and index != len(stack) - 1:
                continue
            if name in bound:
                return True
        return False

    def visit(node: ast.AST, stack: list[tuple[ast.AST, set[str], set[str]]]) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in module_bound and not resolves_local(node.id, stack):
                found.append(node)
            return
        if isinstance(node, _SCOPES):
            outer: list[ast.AST] = []
            inner: list[ast.AST] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                args = node.args
                outer.extend([*args.defaults, *(d for d in args.kw_defaults if d is not None)])
                if not isinstance(node, ast.Lambda):
                    outer.extend(node.decorator_list)
                    all_args = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
                    outer.extend(a.annotation for a in all_args if a is not None and a.annotation is not None)
                    if node.returns is not None:
                        outer.append(node.returns)
                    inner.extend(node.body)
                else:
                    inner.append(node.body)
            elif isinstance(node, ast.ClassDef):
                outer.extend([*node.decorator_list, *node.bases, *(k.value for k in node.keywords)])
                inner.extend(node.body)
            else:
                outer.append(node.generators[0].iter)
                for index, generator in enumerate(node.generators):
                    if index:
                        inner.append(generator.iter)
                    inner.extend(generator.ifs)
                inner.extend([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt])
            for child in outer:
                visit(child, stack)
            bound, declared = _scope_bindings(node)
            stack.append((node, bound, declared))
            for child in inner:
                visit(child, stack)
            stack.pop()
            return
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    visit(tree, [])
    return found


def _dynamic_import_calls(tree: ast.Module) -> list[tuple[ast.Call, str | None]]:
    """Return importlib/__import__ calls and their statically resolved module, if known."""
    importlib_modules: set[str] = set()
    import_module_functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'importlib':
                    importlib_modules.add(alias.asname or 'importlib')
                elif alias.name.startswith('importlib.') and alias.asname is None:
                    importlib_modules.add('importlib')
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'importlib':
            for alias in node.names:
                if alias.name == 'import_module':
                    import_module_functions.add(alias.asname or alias.name)

    found: list[tuple[ast.Call, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        api = ''
        if isinstance(node.func, ast.Name) and node.func.id == '__import__':
            api = '__import__'
        elif isinstance(node.func, ast.Name) and node.func.id in import_module_functions:
            api = 'importlib.import_module'
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id in importlib_modules):
            api = 'importlib.import_module'
        if not api:
            continue

        name_arg = node.args[0] if node.args else next(
            (keyword.value for keyword in node.keywords if keyword.arg == 'name'), None)
        if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
            found.append((node, None))
            continue
        module_name = name_arg.value
        if api == '__import__':
            level_arg = node.args[4] if len(node.args) > 4 else next(
                (keyword.value for keyword in node.keywords if keyword.arg == 'level'), None)
            if level_arg is not None and not (
                    isinstance(level_arg, ast.Constant) and level_arg.value == 0):
                found.append((node, None))
                continue
        if module_name.startswith('.'):
            package_arg = node.args[1] if api == 'importlib.import_module' and len(node.args) > 1 else next(
                (keyword.value for keyword in node.keywords if keyword.arg == 'package'), None)
            if (api != 'importlib.import_module' or not isinstance(package_arg, ast.Constant)
                    or not isinstance(package_arg.value, str)):
                found.append((node, None))
                continue
            package_parts = package_arg.value.split('.')
            level = len(module_name) - len(module_name.lstrip('.'))
            if level > len(package_parts):
                found.append((node, None))
                continue
            base = package_parts[:len(package_parts) - level + 1]
            suffix = module_name[level:].split('.') if module_name[level:] else []
            module_name = '.'.join((*base, *suffix))
        found.append((node, module_name))
    return found


def _dynamic_import_targets_producer(module: str, mapping: dict[str, str],
                                     producer: set[str]) -> bool:
    """An import executes its module and each parent package initializer."""
    parts = module.split('.')
    return any(mapping.get('.'.join(parts[:index])) in producer
               for index in range(1, len(parts) + 1))


def _coupling_for(src: UnitContract, dst: UnitContract, snapshot: Snapshot,
                  mapping: dict[str, str], deadline: float) -> tuple[str, str, tuple[Evidence, ...], str]:
    producer = {path for path in (*src.owns, *src.creates) if snapshot.exists(path)}
    if not producer:
        return 'INDEPENDENT_BY_SCAN', '', (), ''
    names = set()
    for path in producer:
        if time.monotonic() > deadline:
            return 'UNKNOWN', 'TIME_LIMIT', (), ''
        found = _definitions(path, snapshot.read_bytes(path))
        if found is None:
            return 'UNKNOWN', 'UNSUPPORTED_LANGUAGE', (), ''
        names.update(found)
    for path in (*dst.owns, *dst.creates):
        if not snapshot.exists(path):
            continue
        if time.monotonic() > deadline:
            return 'UNKNOWN', 'TIME_LIMIT', (), ''
        data = snapshot.read_bytes(path)
        if path.endswith('.py'):
            try:
                tree = ast.parse(data.decode('utf-8'))
            except (UnicodeError, SyntaxError):
                return 'UNKNOWN', 'UNPARSEABLE_PYTHON', (), ''
            for call, module in _dynamic_import_calls(tree):
                if module is None or _dynamic_import_targets_producer(module, mapping, producer):
                    detail = ('dynamic import with computed name' if module is None
                              else f'dynamic import {module}')
                    return 'COUPLED', '', (Evidence(path, call.lineno, detail),), 'dynamic_import'
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = mapping.get(alias.name)
                        if target in producer or any(key.startswith(alias.name + '.') and value in producer for key, value in mapping.items()):
                            return 'COUPLED', '', (Evidence(path, node.lineno, f'import {alias.name}'),), 'static_import'
                elif isinstance(node, ast.ImportFrom):
                    current = next((name for name, file in mapping.items() if file == path), '')
                    base_parts = (current if path.endswith('/__init__.py') else current.rpartition('.')[0]).split('.')
                    base = '.'.join(base_parts[:len(base_parts) - node.level + 1]) if node.level else ''
                    module = '.'.join(part for part in (base, node.module or '') if part)
                    if mapping.get(module) in producer:
                        return 'COUPLED', '', (Evidence(path, node.lineno, f'from {module} import'),), 'static_import'
                    for alias in node.names:
                        target = mapping.get(f'{module}.{alias.name}')
                        if target in producer:
                            return 'COUPLED', '', (Evidence(path, node.lineno, f'from {module} import {alias.name}'),), 'static_import'
                        package = mapping.get(module)
                        if package and package.endswith('__init__.py'):
                            try:
                                _, defining, _, _ = _resolve_symbol(snapshot, package, alias.name, set())
                            except (OSError, UnicodeError):
                                return 'UNKNOWN', 'AMBIGUOUS_REEXPORT', (), ''
                            if defining in producer:
                                return 'COUPLED', '', (Evidence(path, node.lineno, f'from {module} import {alias.name}'),), 'static_import'
            # An unqualified name is evidence only when it resolves to a global the consumer module never binds.
            for node in sorted(_free_global_loads(tree), key=lambda item: (item.lineno, item.col_offset)):
                if node.id in names:
                    return 'COUPLED', '', (Evidence(path, node.lineno, node.id),), 'static_name'
        else:
            try:
                source = data.decode('utf-8')
            except UnicodeError:
                return 'UNKNOWN', 'UNDECODABLE_SOURCE', (), ''
            if _other_definitions(data, path) is None:
                return 'UNKNOWN', 'UNSUPPORTED_LANGUAGE', (), ''
            for name in sorted(names):
                match = re.search(r'\b' + re.escape(name) + r'\b', source)
                if match:
                    return 'COUPLED', '', (Evidence(path, source.count('\n', 0, match.start()) + 1, name),), 'static_name'
    return 'INDEPENDENT_BY_SCAN', '', (), ''


def scan_coupling(contracts: Sequence[UnitContract], snapshot: Snapshot,
                  limits: ParallelLimits) -> tuple[list[CouplingEdge], list[CouplingPair]]:
    ordered = sorted(contracts, key=lambda item: item.id)
    def unknown(reason: str) -> list[CouplingPair]:
        return [CouplingPair(a.id, b.id, 'UNKNOWN', reason, ())
                for index, a in enumerate(ordered) for b in ordered[index + 1:]]
    paths = sorted(snapshot.list_files())
    if len(paths) > limits.scan_max_files:
        return [], unknown('FILE_LIMIT')
    byte_count = 0
    for path in paths:
        physical = snapshot.root / path
        if physical.is_symlink() or (physical.exists() and not physical.is_file()):
            return [], unknown('UNSAFE_REPOSITORY_ENTRY')
        byte_count += physical.stat().st_size
        if byte_count > limits.scan_max_bytes:
            return [], unknown('BYTE_LIMIT')
    if _ambiguous_modules(snapshot):
        return [], unknown('AMBIGUOUS_MODULE')
    mapping = _module_map(snapshot)
    deadline = time.monotonic() + limits.scan_seconds
    edges: list[CouplingEdge] = []
    pairs: list[CouplingPair] = []
    exports = {export.id: unit.id for unit in contracts for export in unit.exports}
    exports_by_symbol = {(export.path, export.symbol): unit.id
                         for unit in contracts for export in unit.exports}
    def producer(consume: ConsumeSpec) -> str | None:
        if consume.export_id is not None:
            return exports.get(consume.export_id)
        return exports_by_symbol.get((consume.path, consume.symbol))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            declared_lr = any(producer(consume) == left.id for consume in right.consumes)
            declared_rl = any(producer(consume) == right.id for consume in left.consumes)
            status_lr, reason_lr, evidence_lr, source_lr = ('COUPLED', 'DECLARED', (), '') if declared_lr else _coupling_for(left, right, snapshot, mapping, deadline)
            status_rl, reason_rl, evidence_rl, source_rl = ('COUPLED', 'DECLARED', (), '') if declared_rl else _coupling_for(right, left, snapshot, mapping, deadline)
            if status_lr == 'COUPLED' and not declared_lr:
                edges.append(CouplingEdge(left.id, right.id, source_lr, evidence_lr))
            if status_rl == 'COUPLED' and not declared_rl:
                edges.append(CouplingEdge(right.id, left.id, source_rl, evidence_rl))
            status = 'COUPLED' if 'COUPLED' in (status_lr, status_rl) else ('UNKNOWN' if 'UNKNOWN' in (status_lr, status_rl) else 'INDEPENDENT_BY_SCAN')
            pairs.append(CouplingPair(left.id, right.id, status, reason_lr or reason_rl,
                                      evidence_lr + evidence_rl))
    return edges, pairs


def _order(units: Sequence[UnitContract], edges: Sequence[CouplingEdge]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    graph = {unit.id: set() for unit in units}
    indegree = {unit.id: 0 for unit in units}
    for edge in edges:
        if edge.dst_unit not in graph[edge.src_unit]:
            graph[edge.src_unit].add(edge.dst_unit)
            indegree[edge.dst_unit] += 1
    ready = [unit for unit, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        unit = heapq.heappop(ready)
        result.append(unit)
        for other in sorted(graph[unit]):
            indegree[other] -= 1
            if indegree[other] == 0:
                heapq.heappush(ready, other)
    if len(result) == len(graph):
        return tuple(result), ()
    # Tarjan isolates cycle members from descendants blocked by the cycle.
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    members: set[str] = set()
    def visit(vertex: str) -> None:
        nonlocal index
        indices[vertex] = low[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for other in graph[vertex]:
            if other not in indices:
                visit(other)
                low[vertex] = min(low[vertex], low[other])
            elif other in on_stack:
                low[vertex] = min(low[vertex], indices[other])
        if low[vertex] == indices[vertex]:
            group = []
            while True:
                other = stack.pop()
                on_stack.remove(other)
                group.append(other)
                if other == vertex:
                    break
            if len(group) > 1 or vertex in graph[vertex]:
                members.update(group)
    for vertex in graph:
        if vertex not in indices:
            visit(vertex)
    return (), tuple(sorted(members))


def admit(request: ParallelRequest, snapshot: Snapshot, requirements: Sequence[RequirementClause],
          limits: ParallelLimits, *, protected: Callable[[str], bool], case_insensitive: bool) -> AdmissionResult:
    original_units = request.units
    known_requirements = {clause.id for clause in requirements}
    for unit in original_units:
        unknown_ids = set(unit.requirement_ids) - known_requirements
        if unknown_ids:
            return _refuse(snapshot, limits, 'UNKNOWN_REQUIREMENT', ','.join(sorted(unknown_ids)))
    if len(original_units) > limits.max_units:
        return _refuse(snapshot, limits, 'TOO_MANY_UNITS',
                       _limit_detail('max_units', limits.max_units))
    if len({unit.id for unit in original_units}) != len(original_units):
        return _refuse(snapshot, limits, 'UNSAFE_PATH', 'duplicate subframe id')
    units = original_units
    adjustments: tuple[AdmissionAdjustment, ...] = ()
    if len(units) < limits.min_units:
        return _refuse(snapshot, limits, 'TOO_FEW_UNITS',
                       _limit_detail('min_units', limits.min_units))
    owner: dict[str, tuple[str, str]] = {}
    all_paths = []
    paths_by_unit = {
        item.id: (*item.owns, *item.creates, *item.reads, *item.check.source_paths)
        for item in units
    }
    for unit in units:
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,23}', unit.id) or len(unit.outcome) > 300:
            return _refuse(snapshot, limits, 'UNSAFE_PATH', f'invalid subframe identity or outcome: {unit.id}')
        paths = (*unit.owns, *unit.creates)
        if not paths:
            return _refuse(snapshot, limits, 'TOO_MANY_PATHS', unit.id, paths)
        unit_paths = (*paths, *unit.reads)
        if len(unit_paths) > limits.max_paths_per_unit:
            return _refuse(snapshot, limits, 'TOO_MANY_PATHS',
                           _limit_detail('max_paths_per_unit', limits.max_paths_per_unit), unit_paths)
        for path in paths:
            key = path.casefold() if case_insensitive else path
            for prior, (prior_unit, prior_path) in owner.items():
                if key == prior or key.startswith(prior + '/') or prior.startswith(key + '/'):
                    # Identical paths are named once; distinct overlapping paths keep both.
                    return _refuse(snapshot, limits, 'OWNERSHIP_OVERLAP', f'{unit.id} and {prior_unit}',
                                   tuple(dict.fromkeys((path, prior_path))),
                                   members=(unit.id, prior_unit))
            path_reason, missing_ancestors = _path_safety_reason(
                path, snapshot.root, create=path in unit.creates)
            if path_reason is not None or protected(path):
                reason = path_reason or 'path is protected'
                return _refuse(snapshot, limits, 'UNSAFE_PATH', f'{path}: {reason}', (path,))
            if path in unit.creates:
                for ancestor in missing_ancestors:
                    ancestor_key = ancestor.casefold() if case_insensitive else ancestor
                    prefix = ancestor_key + '/'
                    for other in units:
                        if other.id == unit.id:
                            continue
                        for other_path in paths_by_unit[other.id]:
                            if not isinstance(other_path, str):
                                continue
                            other_key = other_path.casefold() if case_insensitive else other_path
                            if other_key == ancestor_key or other_key.startswith(prefix):
                                detail = (f'{path}: missing ancestor {ancestor} is also used by '
                                          f'subframe {other.id} path {other_path}')
                                return _refuse(snapshot, limits, 'UNSAFE_PATH', detail, (path,))
            if path in unit.owns and not snapshot.exists(path):
                return _refuse(snapshot, limits, 'UNSAFE_PATH', f'missing owned file {path}', (path,))
            if path in unit.creates and snapshot.exists(path):
                return _refuse(snapshot, limits, 'CREATE_EXISTS', path, (path,))
            owner[key] = (unit.id, path)
        all_paths.extend(unit_paths)
    if len(all_paths) > limits.max_paths_total:
        return _refuse(snapshot, limits, 'TOO_MANY_PATHS',
                       _limit_detail('max_paths_total', limits.max_paths_total), all_paths)
    for unit in units:
        size = sum((snapshot.root / path).stat().st_size for path in unit.owns)
        if size > limits.max_owned_source_bytes:
            return _refuse(snapshot, limits, 'UNIT_CONTEXT_TOO_LARGE',
                           f'owned_source_bytes={size}; '
                           + _limit_detail('max_owned_source_bytes', limits.max_owned_source_bytes), unit.owns)
        for read in unit.reads:
            key = read.casefold() if case_insensitive else read
            if not _safe_path(read, snapshot.root, create=False) or not snapshot.exists(read) or any(
                    key == owned or key.startswith(owned + '/') or owned.startswith(key + '/')
                    for owned in owner):
                return _refuse(snapshot, limits, 'READ_NOT_ALLOWED', read, (read,))
        for source in unit.check.source_paths:
            key = source.casefold() if case_insensitive else source
            overlaps_other = any(
                (key == owned or key.startswith(owned + '/') or owned.startswith(key + '/'))
                and owner_id != unit.id for owned, (owner_id, _) in owner.items())
            if (not _safe_path(source, snapshot.root, create=False) or
                    not snapshot.exists(source) or overlaps_other):
                return _refuse(snapshot, limits, 'READ_NOT_ALLOWED', source, (source,))
    exports: dict[str, tuple[str, str, str]] = {}
    export_paths: dict[tuple[str, str], str] = {}
    for unit in units:
        for export in unit.exports:
            if export.path not in (*unit.owns, *unit.creates):
                return _refuse(snapshot, limits, 'EXPORT_OUTSIDE_OWNERSHIP', export.id, (export.path,))
            earlier = exports.get(export.id)
            earlier_id = export_paths.get((export.path, export.symbol))
            if earlier is not None or earlier_id is not None:
                detail = (f'{earlier_id} and {export.id} name {export.path}::{export.symbol}'
                          if earlier_id is not None else f'{export.id} is declared more than once')
                return _refuse(snapshot, limits, 'DUPLICATE_EXPORT', detail, (export.path,))
            exports[export.id] = (unit.id, export.path, export.symbol)
            export_paths[(export.path, export.symbol)] = export.id
    edges = []
    for unit in units:
        for consume in unit.consumes:
            if consume.export_id is not None:
                target = exports.get(consume.export_id)
                if target is None:
                    return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME', consume.export_id)
                if consume.path is not None and (consume.path, consume.symbol) != target[1:]:
                    return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME',
                                   f'{consume.export_id} declares {target[1]}::{target[2]}, '
                                   f'but consume names {consume.path}::{consume.symbol}',
                                   (target[1], consume.path))
                if target[0] == unit.id:
                    return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME', 'self consume')
                if not target[1].endswith('.py') and (
                        unit.check.command_kind not in ('pytest', 'run_check_command') or
                        not unit.check.selectors or any(not isinstance(selector, str) or not selector.strip()
                                                        for selector in unit.check.selectors)):
                    return _refuse(snapshot, limits, 'UNSUPPORTED_CONTRACT', consume.export_id, (target[1],))
                edges.append(CouplingEdge(target[0], unit.id, 'declared', ()))
            else:
                path = consume.path or ''
                symbol = consume.symbol or ''
                export_id = export_paths.get((path, symbol))
                if export_id is not None:
                    target = exports[export_id]
                    if target[0] == unit.id:
                        return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME', 'self consume', (path,))
                    if not path.endswith('.py') and (
                            unit.check.command_kind not in ('pytest', 'run_check_command') or
                            not unit.check.selectors):
                        return _refuse(snapshot, limits, 'UNSUPPORTED_CONTRACT', export_id, (path,))
                    edges.append(CouplingEdge(target[0], unit.id, 'declared', ()))
                    continue
                key = path.casefold() if case_insensitive else path
                if not _safe_path(path, snapshot.root, create=False) or not snapshot.exists(path) or any(
                        key == owned or key.startswith(owned + '/') or owned.startswith(key + '/') for owned in owner):
                    return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME', symbol, (path,))
                if path.endswith('.py'):
                    projection = project_interface(snapshot, path, symbol, None)
                    if not projection.line:
                        try:
                            tree = ast.parse(snapshot.read_bytes(path).decode('utf-8'))
                            matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol]
                        except (SyntaxError, UnicodeError):
                            matches = []
                        reason = 'AMBIGUOUS_CONSUME' if len(matches) > 1 else 'UNRESOLVED_CONSUME'
                        return _refuse(snapshot, limits, reason, symbol, (path,))
                    if projection.signature is None:
                        return _refuse(snapshot, limits, 'AMBIGUOUS_CONSUME', symbol, (path,))
                else:
                    if (snapshot.root / path).stat().st_size > limits.scan_max_bytes:
                        return _refuse(snapshot, limits, 'AMBIGUOUS_CONSUME', symbol, (path,))
                    data = snapshot.read_bytes(path)
                    definitions = _other_definitions(data, path)
                    if definitions is None or len(data) > limits.scan_max_bytes:
                        return _refuse(snapshot, limits, 'AMBIGUOUS_CONSUME', symbol, (path,))
                    count = sum(1 for name, _ in definitions if name == symbol)
                    if count == 0:
                        return _refuse(snapshot, limits, 'UNRESOLVED_CONSUME', symbol, (path,))
                    if count > 1:
                        return _refuse(snapshot, limits, 'AMBIGUOUS_CONSUME', symbol, (path,))
    scan_edges, pairs = scan_coupling(units, snapshot, limits)
    if scan_edges:
        ordered_edges = sorted(scan_edges, key=lambda edge: (edge.src_unit, edge.dst_unit))
        hit = scan_edges[0]
        ordered_edges.remove(hit)
        ordered_edges.insert(0, hit)
        return _refuse(snapshot, limits, 'UNDECLARED_COUPLING',
                       ', '.join(f'{edge.src_unit}->{edge.dst_unit}' for edge in ordered_edges),
                       sorted({edge.evidence[0].path for edge in ordered_edges if edge.evidence}),
                       [item for edge in ordered_edges for item in edge.evidence])
    order, cycle = _order(units, edges)
    if cycle:
        return _refuse(snapshot, limits, 'DEPENDENCY_CYCLE', ','.join(cycle), members=cycle)
    return AdmissionResult('ADMITTED', None,
                           tuple(sorted(edges, key=lambda edge: (edge.src_unit, edge.dst_unit))),
                           tuple(pairs), order, snapshot.generation, snapshot.manifest_digest,
                           limits, tuple(units), adjustments)


def rescan_proposal(contracts: Sequence[UnitContract], admission: AdmissionResult,
                    composed: Snapshot, unit_id: str) -> RescanResult:
    if unit_id not in {unit.id for unit in contracts}:
        raise ValueError(f'unknown subframe {unit_id!r}')
    edges, pairs = scan_coupling(contracts, composed, admission.limits)
    if edges:
        hit = edges[0]
        return RescanResult(Refusal('UNDECLARED_COUPLING', f'{hit.src_unit}->{hit.dst_unit}',
                                    (hit.evidence[0].path,), hit.evidence, ()), tuple(pairs))
    return RescanResult(None, tuple(pairs))
