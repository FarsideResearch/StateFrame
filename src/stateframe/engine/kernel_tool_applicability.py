"""Conservative dispatch facts shared by the frame and optional tool filter."""
from __future__ import annotations

import ast
import hashlib
from pathlib import PurePosixPath


def only_target_product_module(files: dict[str, str], target: str) -> bool:
    """Do not hide a path-specific editor when other editable product files exist."""
    empty = 'sha256:' + hashlib.sha256(b'').hexdigest()
    return target in files and all(path == target or
        (PurePosixPath(path).name == '__init__.py' and revision == empty)
        for path, revision in files.items())


def target_kind(source: str, symbol: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 'currently_invalid_python'
    if '.' in symbol:
        def qualified(node: ast.AST, prefix: tuple[str, ...] = ()) -> bool:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = (*prefix, node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and '.'.join(prefix) == symbol:
                return True
            return any(qualified(child, prefix) for child in ast.iter_child_nodes(node))
        if qualified(tree):
            return 'qualified_function'
    target = next((node for node in tree.body if getattr(node, 'name', None) == symbol), None)
    if isinstance(target, ast.ClassDef):
        return 'class'
    if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 'top_level_function'
    return 'unknown'


def edit_applicability(source: str, symbol: str, *, multi_file: bool) -> dict:
    kind = target_kind(source, symbol)
    finding = {'target_kind': kind, 'remove_tools': [], 'reason': ''}
    # The single-file edit dispatcher accepts complete top-level classes as well as functions.
    # Preserve the old conservative result for every other target shape.
    return finding
