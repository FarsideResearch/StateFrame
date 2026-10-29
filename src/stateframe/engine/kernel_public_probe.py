"""Public-only test proposals are fallible evidence, never completion authority."""
from __future__ import annotations

import ast
import json
from pathlib import PurePosixPath

SCHEMA = {
    'type': 'object', 'properties': {
        'test_source': {'type': 'string'},
        'requirement_quotes': {'type': 'array', 'items': {'type': 'string'}},
        'uncertainties': {'type': 'array', 'items': {'type': 'string'}},
    }, 'required': ['test_source', 'requirement_quotes', 'uncertainties'],
    'additionalProperties': False,
}
OUTPUT_CONTRACT = (
    'Output contract: return exactly one JSON object with three fields. '
    'test_source is a string containing complete executable Python source: imports and one '
    'full def test_... function with its body. Never return just a function name, path, summary '
    'or reference to code elsewhere. requirement_quotes is a list of exact substrings copied '
    'from the public task that support the asserted outcomes; do not add quote characters '
    'inside those strings. uncertainties is a list of unresolved assumptions, or [] if none. '
    'If no justified probe remains, return an empty test_source string. Do not invent a test '
    'merely to fill the field. Return JSON only, without markdown fences.'
)
DESIGN = (
    'You are an independent public-contract test designer. Propose one self-contained '
    'pytest test function checking an important stated behavior or interaction not already '
    'covered by the supplied tests. Use only the public request and original interface. '
    'Do not implement the product or invent requirements. Include imports from app or pytest '
    'and deterministic inputs. Quote the exact requirement clauses supporting expectations. '
    'Do not use filesystem, network, clock, subprocess, dynamic code, skips or xfails. '
    'List ambiguity as uncertainty instead of asserting a guessed answer. '
    'A passing generated test never proves completeness. '
) + OUTPUT_CONTRACT
REVIEW = (
    'Independently audit the proposed test against the public contract. Check every '
    'expectation, scenario and import. Remove unsupported or ambiguous assertions; correct '
    'mistakes only when the public requirement supports the correction. You have not seen '
    'candidate code or evaluation results. Copy supporting clauses verbatim from the public request; '
    'do not add quotation marks inside the quote strings. Return the same JSON schema. Return an empty '
    'test_source if no justified probe remains. Preserve the output contract: test_source '
    'must contain the complete corrected executable source, never merely its name.'
)


def public_bundle(row: dict) -> dict:
    """Explicit allowlist; never pass an entire worker row to a reviewer."""
    path = PurePosixPath(str(row.get('rel', '')))
    if (row.get('repo') != 'build' or path.is_absolute() or '..' in path.parts
            or path.suffix != '.py' or any(not part.isidentifier() for part in path.with_suffix('').parts)):
        raise ValueError('public review currently supports declared Python build interfaces only')
    fields = {'task': row.get('msg'), 'public_interface_source': row.get('parent_src'),
              'existing_public_tests': row.get('test_src'),
              'public_module': 'app.' + '.'.join(path.with_suffix('').parts)}
    if row.get('files') is not None:
        from kernel_public_multifile import bundle
        fields = bundle(row)
    if any(not isinstance(fields[key], str) or not fields[key].strip()
           for key in ('task', 'public_interface_source', 'existing_public_tests', 'public_module')):
        raise ValueError('public review requires explicit public brief, interface and tests')
    if len(json.dumps(fields).encode()) > 24000:
        raise ValueError('public review input exceeds bounded context')
    return fields


def validate_draft(value: object) -> dict:
    """Bound untrusted review input; this never authorizes execution."""
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('invalid public probe fields')
    source = value['test_source']
    if not isinstance(source, str) or len(source.encode()) > 8000:
        raise ValueError('missing or oversized public probe')
    for field in ('requirement_quotes', 'uncertainties'):
        items = value[field]
        if (not isinstance(items, list) or len(items) > 8
                or any(not isinstance(item, str) or not 0 < len(item) <= 2000 for item in items)):
            raise ValueError('invalid quote or uncertainty list')
    return json.loads(json.dumps(value))


def validate_proposal(value: object, brief: str, *, public_module: str | None = None,
                      max_tests: int = 1, public_modules: list[str] | None = None) -> dict:
    if type(max_tests) is not int or not 1 <= max_tests <= 8:
        raise ValueError('public probe test bound must be 1..8')
    allowed_modules = None
    if public_modules is not None:
        if (not isinstance(public_modules, list) or not 1 <= len(public_modules) <= 100
                or any(not isinstance(name, str) or not (name == 'app' or name.startswith('app.'))
                       or any(not part.isidentifier() for part in name.split('.'))
                       for name in public_modules)
                or len(set(public_modules)) != len(public_modules)
                or public_module not in public_modules):
            raise ValueError('invalid declared public module allowlist')
        allowed_modules = set(public_modules)
    elif public_module is not None:
        allowed_modules = {public_module}
    value = validate_draft(value)
    source = value['test_source']
    if not source.strip():
        raise ValueError('missing public probe')
    if not value['requirement_quotes'] or any(quote not in brief for quote in value['requirement_quotes']):
        raise ValueError('probe expectations lack exact public requirement quotes')
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if (not 1 <= len(functions) <= max_tests
            or len({function.name for function in functions}) != len(functions)
            or any(not function.name.startswith('test_') or function.decorator_list
                   or function.args.args or function.args.kwonlyargs or function.args.vararg
                   or function.args.kwarg or function.args.posonlyargs for function in functions)
            or any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body)):
        raise ValueError('probe must contain imports and bounded unique undecorated fixture-free tests')
    pytest_names = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.Import)
                    for alias in node.names if alias.name == 'pytest'}
    raises_names = {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                    and node.module == 'pytest' for alias in node.names if alias.name == 'raises'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None or any(alias.name == '*' for alias in node.names):
                raise ValueError('relative or wildcard probe imports are unavailable')
            names = [node.module]
            if node.module == 'pytest' and any(alias.name not in {'raises', 'approx'} for alias in node.names):
                raise ValueError('unsupported pytest import')
        else:
            names = []
        if any(name != 'pytest' and not name.startswith('app.')
               and not (name == 'app' and allowed_modules is not None and name in allowed_modules)
               for name in names):
            raise ValueError('probe import outside public product/pytest')
        if allowed_modules is not None and any(name not in {'pytest', *allowed_modules} for name in names):
            raise ValueError('probe import outside declared public interface')
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.While, ast.Global, ast.Nonlocal)):
            raise ValueError('unsupported probe construct')
        if isinstance(node, ast.FunctionDef) and node not in functions:
            raise ValueError('nested probe functions are unavailable')
        if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in {
                'open', 'eval', 'exec', 'compile', 'globals', 'locals', 'vars', 'getattr',
                'setattr', 'delattr', 'input', 'print', 'breakpoint', 'exit', 'quit'}):
            raise ValueError('dynamic or external probe operation')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in {
                'skip', 'xfail', 'mark', 'main', 'exit', 'importorskip', 'MonkeyPatch'}):
            raise ValueError('probe cannot bypass test execution')
    for function in functions:
        assertions = 0
        for node in ast.walk(function):
            assertions += isinstance(node, ast.Assert)
            if isinstance(node, ast.With):
                assertions += sum(isinstance(item.context_expr, ast.Call) and (
                    isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id in raises_names
                    or isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == 'raises'
                    and isinstance(item.context_expr.func.value, ast.Name)
                    and item.context_expr.func.value.id in pytest_names) for item in node.items)
        if not assertions:
            raise ValueError('every probe test must contain an assertion')
    # Syntax restrictions reduce accidental hazards; the OS sandbox is the security boundary.
    return json.loads(json.dumps(value))
