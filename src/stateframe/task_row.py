"""Turn a working copy plus a task into the engine's multi-file task row.

The row has the same shape the engine receives on its live release path (a "multi_file"
build row): every text file of the repository as declared product source, the public check
files separately, a ticket (``msg``), and a starting-file hint (``entry``). The engine runs
the public checks with pytest inside its own sandboxed copy.
"""
from __future__ import annotations

import ast
import hashlib
import posixpath
import re
from pathlib import Path, PurePosixPath

from .workcopy import WorkCopy

TEXT_SUFFIXES = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
    ".csv", ".sql", ".html", ".css", ".js", ".ts", ".sh", ".in",
}
TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")


class TaskRowError(ValueError):
    pass


def discover_checks(files: dict[str, str]) -> list[str]:
    """pytest-style test files in the repository, sorted."""
    return sorted(path for path in files if TEST_FILE.search(path))


def _imported_modules(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def guess_entry(product: dict[str, str], checks: dict[str, str]) -> str | None:
    """First product module the checks import; a starting-file hint, never a constraint."""
    by_module: dict[str, str] = {}
    for path in product:
        if not path.endswith(".py"):
            continue
        parts = list(PurePosixPath(path).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):  # also match with a src/ or package prefix stripped
            by_module.setdefault(".".join(parts[start:]), path)
    for check in sorted(checks):
        for module in _imported_modules(checks[check]):
            if module in by_module:
                return by_module[module]
            head = module.split(".")[0]
            if head in by_module and not by_module[head].endswith("__init__.py"):
                return by_module[head]
    python = sorted(p for p in product if p.endswith(".py") and not p.endswith("__init__.py"))
    if python:
        return python[0]
    # Non-Python repos (a static site, a create-from-scratch folder): start at the first file.
    return sorted(product)[0] if product else None


# Above this many product files the engine is not handed every file as editable source: the
# frame lists editable paths, and a whole large repo there drowns the state. The rest of the repo
# is still present in the engine's sandbox as read-only context (see _child.py).
DECLARE_ALL_LIMIT = 200


def repo_relative(path: str) -> str:
    """A repository path as the file map spells it: `./tests/x.py` and `tests//x.py` mean `tests/x.py`."""
    text = path.replace("\\", "/")
    normal = posixpath.normpath(text) if text else text
    if not normal or normal.startswith("/") or normal == ".." or normal.startswith("../"):
        raise TaskRowError(f"path must be inside the repository and relative to its root: {path}")
    return normal


def build_row(work: WorkCopy, *, task: str, checks: list[str] | None = None,
              target: str | None = None, task_id: str | None = None,
              edit_paths: list[str] | None = None) -> dict[str, object]:
    texts = {path: text for path, text in work.text_files().items()
             if Path(path).suffix in TEXT_SUFFIXES or Path(path).name in {"Makefile", "setup.py"}}
    if not task.strip():
        raise TaskRowError("--task is empty")
    checks = [repo_relative(path) for path in checks or []]
    target = repo_relative(target) if target else target
    edit_paths = [repo_relative(path) for path in edit_paths or []] or edit_paths
    declared_checks = checks if checks else discover_checks(texts)
    if not declared_checks:
        raise TaskRowError("no test files found; pass --check <path> for at least one pytest file")
    missing = [path for path in declared_checks if path not in texts]
    if missing:
        raise TaskRowError("check file(s) not found in the repository: " + ", ".join(missing))
    public_tests = {path: texts[path] for path in declared_checks}
    all_product = {path: text for path, text in texts.items()
                   if path not in public_tests and not path.startswith("tests/")}
    if target is not None:
        target = target.strip()
        if target not in all_product:
            raise TaskRowError(f"--target {target} is not a product source file in the repository")
    edits = [p.strip() for p in (edit_paths or [])]
    bad = [p for p in edits if p not in all_product]
    if bad:
        raise TaskRowError("--edit file(s) are not product source files in the repository: "
                           + ", ".join(bad))
    if edits or len(all_product) > DECLARE_ALL_LIMIT:
        # Scoped mode: only these files are declared editable; the whole repository is still in
        # the sandbox as context. Other tests stay in the repository copy, not the row.
        scope = list(dict.fromkeys(([target] if target else []) + edits))
        if not scope:
            raise TaskRowError(f"the repository has {len(all_product)} source files; pass --target "
                               "(and --edit for any other file the fix may touch)")
        product = {path: all_product[path] for path in scope}
    else:
        product = all_product
        # Other tests under tests/ (fixtures, conftest, helpers) stay visible as public test files.
        for path, text in texts.items():
            if path.startswith("tests/") and path not in public_tests:
                public_tests[path] = text
    if not product:
        raise TaskRowError("repository has no product source files outside its tests")
    entry = target or guess_entry(product, {p: public_tests[p] for p in declared_checks})
    if entry is None:
        raise TaskRowError("could not pick a starting file; pass --target <path>")
    # Product roots: each top-level directory, plus each top-level file by name (the engine
    # matches a root either exactly or as a directory prefix; "." would match nothing).
    roots = sorted({path.split("/", 1)[0] for path in product})
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:12]
    identifier = task_id or f"task-{digest}"
    first_check = declared_checks[0]
    return {
        "task_id": identifier,
        "sha": identifier,
        "repo": "build",
        "shape": "multi_file",
        "task_kind": "bug",
        "tier": "USER-REPO",
        "msg": task.strip(),
        "files": {path: {"parent_src": text} for path, text in sorted(product.items())},
        "scoped_edits": bool(edits or len(all_product) > DECLARE_ALL_LIMIT),
        "public_tests": dict(sorted(public_tests.items())),
        "public_check_paths": list(declared_checks),
        "allowed_product_paths": roots,
        "new_file_roots": [],
        "entry": {"fn": "", "path": entry},
        "path": entry,
        "rel": entry,
        "fn": "",
        "test_path": first_check,
        "test_src": public_tests[first_check],
        "_skip_db": True,
    }
