"""Opt-in declared multi-file input for the existing public review organ."""
from __future__ import annotations
from collections.abc import Mapping
import os
from pathlib import PurePosixPath
from kernel_live_projection import is_harness_path


_FORBIDDEN_PUBLIC_PATH_PARTS = frozenset({
    'private', 'hidden', 'hidden_tests', 'near_miss', 'gold', 'oracle',
    'grader', 'graders', 'grading', 'harness',
})


def enabled() -> bool:
    value = os.environ.get('KERNEL_PUBLIC_REVIEW_MULTIFILE', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_PUBLIC_REVIEW_MULTIFILE must be 0 or 1')
    return value == '1'


def _path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError('public review needs an explicit workspace path')
    path = PurePosixPath(value)
    if (path.is_absolute() or '..' in path.parts or path.as_posix() != value
            or (path.parts and ':' in path.parts[0])
            or path.suffix != '.py'
            or any(part.lower() in _FORBIDDEN_PUBLIC_PATH_PARTS
                   or 'backup' in part.lower() for part in path.parts)
            or any(not part.isidentifier() for part in path.with_suffix('').parts)):
        raise ValueError('invalid public review workspace path')
    return path


def bundle(row: dict) -> dict:
    if not enabled():
        raise ValueError('multi-file public review is disabled')
    if row.get('repo') != 'build':
        raise ValueError('public review requires a declared build workspace')
    files = row.get('files')
    if not isinstance(files, Mapping) or not 1 <= len(files) <= 100:
        raise ValueError('public review requires bounded declared files')
    path = _path(row.get('path'))
    if is_harness_path(path):
        raise ValueError('primary review path must be a public product module')
    checks = row.get('public_check_paths')
    if (not isinstance(checks, list) or not checks or len(checks) > 100
            or any(not isinstance(item, str) for item in checks)
            or len(set(checks)) != len(checks) or str(path) in checks):
        raise ValueError('public review requires explicit distinct public test paths')

    def original(name: str) -> str:
        payload = files.get(name)
        source = payload.get('parent_src') if isinstance(payload, Mapping) else None
        if not isinstance(source, str):
            raise ValueError('public review requires declared original public source')
        return source

    tests = []
    for check in checks:
        if not is_harness_path(_path(check)) or not original(check).strip():
            raise ValueError('public check must be a declared nonempty harness file')
        tests.append('# Public acceptance file: ' + check + '\n' + original(check))
    interfaces = {}
    for name in sorted(files):
        declared = _path(name)
        if is_harness_path(declared):
            continue
        parts = declared.with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        if 'app' in parts:
            parts = parts[parts.index('app'):]
        module = '.'.join(parts)
        if module in interfaces:
            raise ValueError('ambiguous declared public module')
        interfaces[module] = original(name)
    parts = path.with_suffix('').parts
    if 'app' in parts:
        parts = parts[parts.index('app'):]
    primary = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
    return {'task': row.get('msg'), 'public_interface_source': original(str(path)),
            'existing_public_tests': '\n\n'.join(tests),
            'public_module': primary, 'public_modules': sorted(interfaces),
            'public_interface_sources': interfaces}
