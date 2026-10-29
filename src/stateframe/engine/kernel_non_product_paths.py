"""ONE authority for tool output that is never product source (audit 2026-10-05 #2).

Coverage, caches and build output written into the solver root by the model's
own checks (AHA's pytest.ini forces ``--cov ... --cov-report=html``) were bound
as ``product_files``, retained as changed source and journaled until the source
archive hit its 64 MB cap and killed a 10 000 s run.  Every consumer that
enumerates the workspace (live projection binding, retained-context refresh,
environment observation) imports this module; the stage6 host harness keeps a
mirrored literal (it cannot import the kernel) guarded by a drift test.

Language-generic: Python, JS/TS, Rust, JVM, Ruby/PHP/Go vendor caches.
"""
from __future__ import annotations

from pathlib import PurePath, PurePosixPath

# Any path component equal to one of these is never product.
NON_PRODUCT_PARTS = frozenset({
    ".git", ".test-tmp",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis",
    ".tox", ".nox", "htmlcov", ".nyc_output",
    ".gradle",
    "private", "hidden", "hidden_tests", "oracle", "gold", "near_miss",
    "backup", "backups",
})
# Exact basenames (and the ``.coverage.<host>.<pid>`` parallel-data family).
NON_PRODUCT_NAMES = frozenset({".coverage", "coverage.xml", "lcov.info", "coverage-final.json"})
_NAME_PREFIXES = (".coverage.",)
# Generic names only treated as output at the workspace ROOT (a nested
# ``coverage``/``target`` package can be real product source).
ROOT_ONLY_PARTS = frozenset({"coverage", "target"})
# Adjacent component pairs: cache directories inside dependency trees.
NON_PRODUCT_PAIRS = frozenset({
    ("node_modules", ".cache"), ("vendor", "cache"), ("vendor", "bundle"),
})


def is_non_product(relative: PurePath | str) -> bool:
    """True when a workspace-relative path is tool output, not product source."""
    path = PurePosixPath(str(relative).replace("\\", "/"))
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return False
    if any(part.startswith(".") or part in NON_PRODUCT_PARTS for part in parts):
        return True
    if len(parts) > 1 and parts[0] in ROOT_ONLY_PARTS:
        return True
    name = parts[-1]
    if name in NON_PRODUCT_NAMES or name.startswith(_NAME_PREFIXES):
        return True
    return any(pair in NON_PRODUCT_PAIRS for pair in zip(parts, parts[1:]))
