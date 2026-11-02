"""Turn a working copy plus a task into the engine's multi-file task row.

The row has the same shape the engine receives on its live release path (a "multi_file"
build row): every text file of the repository as declared product source, the public check
files separately, a ticket (``msg``), and a starting-file hint (``entry``). Pytest rows
carry public file selectors; generic rows carry one whole-command check.
"""
from __future__ import annotations

import ast
import hashlib
import posixpath
import re
from pathlib import Path, PurePosixPath

from .workcopy import WorkCopy
from .test_runners import is_test_file, judging_test_paths, protected_paths, validate_patterns

TEXT_SUFFIXES = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
    ".csv", ".sql", ".html", ".css", ".js", ".ts", ".sh", ".in",
}
TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
TASK_PATH = re.compile(r"(?<![\w./-])(?:[\w.-]+/)+[\w.-]*")
TASK_ACTION = re.compile(r"\b(build|create|write|produce|save|generate|add|read|load|use)\b",
                         re.IGNORECASE)
NONPUBLIC_PARTS = frozenset({
    "test", "tests", "testdata", "fixtures", "fixture", "private", "hidden",
    "hidden_tests", "oracle", "gold", "grader", "reference",
})


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


def _module_index(product: dict[str, str]) -> dict[str, str]:
    """Dotted module name -> product path, also with a src/ or package prefix stripped."""
    by_module: dict[str, str] = {}
    for path in product:  # first path wins, in the caller's order (guess_entry depends on it)
        if not path.endswith(".py"):
            continue
        parts = list(PurePosixPath(path).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for start in range(len(parts)):
            by_module.setdefault(".".join(parts[start:]), path)
    return by_module


def guess_entry(product: dict[str, str], checks: dict[str, str]) -> str | None:
    """First product module the checks import; a starting-file hint, never a constraint."""
    by_module = _module_index(product)
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


def _public_hint(path: str) -> bool:
    parsed = PurePosixPath(path.replace("\\", "/"))
    parts = parsed.parts
    if parsed.is_absolute() or ".." in parts or not parts or any(
            part.startswith((".", "_")) or part.casefold() in NONPUBLIC_PARTS
                        for part in parts[:-1]):
        return False
    name = parts[-1].casefold()
    return not (name.startswith((".", "_")) and name != "__init__.py"
                or name.startswith("test_") or name.endswith("_test.py")
                or name in {"test.py", "conftest.py"})


def _output_hint(task: str, product: dict[str, str]) -> str | None:
    """Prefer an existing file beside a named output; permit a new named output."""
    named = []
    for match in TASK_PATH.finditer(task):
        actions = list(TASK_ACTION.finditer(task, 0, match.start()))
        action = actions[-1].group().casefold() if actions else ""
        if action not in {"read", "load", "use"}:
            named.append(match.group().rstrip(".,;:"))
    for raw in named:
        path = raw.rstrip("/")
        if not _public_hint(path):
            continue
        parent = path if raw.endswith("/") else posixpath.dirname(path)
        readme = f"{parent}/README.md" if parent else "README.md"
        if readme in product and _public_hint(readme):
            return readme
    for raw in named:
        path = raw.rstrip("/")
        if _public_hint(path) and "." in PurePosixPath(path).name:
            return path
    return None


AUTO_SCOPE_LIMIT = 12
AUTO_SCOPE_CHECK_LIMIT = 8


def auto_scope(task: str, product: dict[str, str], checks: dict[str, str]) -> list[str]:
    """Editable files chosen for an over-limit repository when no --target/--edit was given.

    Product files the task names, then product modules the given checks import, then one more
    import hop from those, at most AUTO_SCOPE_LIMIT in that order. An empty result leaves the
    caller's refusal in place. Deterministic: the same repository and task give the same scope.
    """
    chosen: list[str] = []

    def add(path: str) -> None:
        if path in product and path not in chosen and len(chosen) < AUTO_SCOPE_LIMIT:
            chosen.append(path)

    for match in TASK_PATH.finditer(task):
        add(match.group().rstrip(".,;:/"))
    if len(checks) > AUTO_SCOPE_CHECK_LIMIT:
        # A broad selector (`pytest tests/`) says nothing about where the fix lives.
        return chosen
    by_module = _module_index(product)

    def imported(source: str) -> list[str]:
        paths = []
        for module in _imported_modules(source):
            path = by_module.get(module)
            if path is None:  # as guess_entry: fall back to the top package's module, never its __init__
                head = by_module.get(module.split(".")[0])
                path = head if head is not None and not head.endswith("__init__.py") else None
            if path is not None:
                paths.append(path)
        return paths

    first_hop = [path for check in sorted(checks) for path in imported(checks[check])]
    for path in first_hop:
        add(path)
    for path in list(dict.fromkeys(first_hop)):
        for nested in imported(product[path]):
            add(nested)
    return chosen


# Above this many product files the engine is not handed every file as editable source: the
# frame lists editable paths, and a whole large repo there drowns the state. The rest of the repo
# is still present in the engine's sandbox as read-only context (see _child.py).
DECLARE_ALL_LIMIT = 200


# Opt-in edit scope widening (STATEFRAME_EDIT_SCOPE_WIDEN / --edit-scope widen) admits at most
# this many existing product files beyond a scoped run's declared files. The engine enforces the
# same cap; the host re-checks every widened path before writing the candidate.
EDIT_SCOPE_WIDEN_CAP = 24
_NON_PRODUCT_PARTS = frozenset({"__pycache__", ".pytest_cache", ".git", ".test-tmp", "tests", "test",
                                "probes", "scratch", "tmp", "private", "hidden_tests"})


def widened_scope_paths(row: dict, candidate_paths, baseline, *, hook_paths=()) -> list[str]:
    """Existing product files outside a scoped row's files that edit scope widening may write.

    Host-side check, independent of the engine: the path existed in the repository, is not a
    test or protected file. ``candidate_paths`` is in admission order.
    """
    if row.get("scoped_edits") is not True:
        return []
    declared = set(row.get("files") or ())
    excluded = (set(row.get("protected_test_paths") or ())
                | set(row.get("test_file_paths") or ())
                | set(row.get("public_check_paths") or ())
                | set(row.get("public_tests") or ()) | set(hook_paths))
    chosen = []
    for path in dict.fromkeys(candidate_paths):
        if (path in declared or path not in baseline or path in excluded
                or is_test_file(path) or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts):
            continue
        parts = PurePosixPath(path).parts
        name = parts[-1] if parts else ""
        if (not parts or any(part in _NON_PRODUCT_PARTS for part in parts)
                or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"):
            continue
        chosen.append(path)
    return chosen[:EDIT_SCOPE_WIDEN_CAP]


def repo_relative(path: str) -> str:
    """A repository path as the file map spells it: `./tests/x.py` and `tests//x.py` mean `tests/x.py`."""
    text = path.replace("\\", "/")
    normal = posixpath.normpath(text) if text else text
    if not normal or normal.startswith("/") or normal == ".." or normal.startswith("../"):
        raise TaskRowError(f"path must be inside the repository and relative to its root: {path}")
    return normal


def build_row(work: WorkCopy, *, task: str, checks: list[str] | None = None,
              target: str | None = None, task_id: str | None = None,
              edit_paths: list[str] | None = None, runner_kind: str = "pytest",
              test_command: tuple[str, ...] = (),
              protected_test_globs: tuple[str, ...] = (),
              protect_tests: str = "all") -> dict[str, object]:
    if runner_kind == "generic":
        return _build_generic_row(
            work, task=task, checks=checks, target=target, task_id=task_id,
            edit_paths=edit_paths, test_command=test_command,
            protected_test_globs=protected_test_globs, protect_tests=protect_tests)
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
    test_files = {path for path in texts if is_test_file(path)}
    if protect_tests == "all":
        locked_tests = test_files
    elif protect_tests == "task":
        command_tests = set(judging_test_paths(tuple(test_files), test_command, ()))
        # A command without an explicit file can select any test via discovery or -k.
        if test_command and not command_tests:
            command_tests = test_files
        locked_tests = set(declared_checks) | command_tests
    else:
        locked_tests = set()
    all_product = {path: text for path, text in texts.items()
                   if (path not in public_tests and not path.startswith("tests/"))
                   or (protect_tests != "all" and path in test_files and path not in locked_tests)}
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
            # Only explicitly chosen checks steer the scope: discovery lists every test in the repo.
            steering = set(checks or ()) | set(judging_test_paths(tuple(test_files), test_command, ()))
            scope = auto_scope(task, all_product, {path: texts[path] for path in sorted(steering)
                                                   if path in texts})
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
    entry = target or guess_entry(product, {p: public_tests[p] for p in declared_checks})
    if entry is None:
        entry = _output_hint(task, product)
    if entry is not None and not _public_hint(entry):
        entry = _output_hint(task, product)
        if entry is None:
            entry = guess_entry({path: text for path, text in product.items()
                                 if _public_hint(path)}, {})
    if entry is None:
        raise TaskRowError("could not pick a public starting file; name an output path in --task "
                           "or pass --target for an existing product file")
    new_roots = [entry.split("/", 1)[0]] if entry not in product else []
    # Product roots: each top-level directory, plus each top-level file by name (the engine
    # matches a root either exactly or as a directory prefix; "." would match nothing).
    roots = sorted({*(path.split("/", 1)[0] for path in product), *new_roots})
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:12]
    identifier = task_id or f"task-{digest}"
    first_check = declared_checks[0]
    row = {
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
        "new_file_roots": new_roots,
        "entry": {"fn": "", "path": entry},
        "path": entry,
        "rel": entry,
        "fn": "",
        "test_path": first_check,
        "test_src": public_tests[first_check],
        "_skip_db": True,
    }
    if protect_tests != "all":
        row.update(protect_tests=protect_tests,
                   test_file_paths=sorted(test_files),
                   protected_test_paths=sorted(locked_tests))
    return row


def _build_generic_row(work: WorkCopy, *, task: str, checks: list[str] | None,
                       target: str | None, task_id: str | None,
                       edit_paths: list[str] | None, test_command: tuple[str, ...],
                       protected_test_globs: tuple[str, ...],
                       protect_tests: str) -> dict[str, object]:
    if not task.strip():
        raise TaskRowError("--task is empty")
    patterns = validate_patterns(list(protected_test_globs))
    texts = {path: text for path, text in work.text_files().items()
             if Path(path).suffix in TEXT_SUFFIXES | {
                 ".jsx", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".go", ".rs",
                 ".c", ".h", ".cc", ".cpp", ".hpp",
             } or Path(path).name in {"Makefile", "makefile", "GNUmakefile", "Cargo.toml"}}
    declared = [repo_relative(path) for path in checks or []]
    missing = [path for path in declared if path not in work.baseline]
    if missing:
        raise TaskRowError("check file(s) not found in the repository: " + ", ".join(missing))
    all_protected = set(protected_paths(
        tuple(work.baseline), kind="generic", extra_patterns=patterns,
        command=test_command)) | set(declared)
    test_files = {path for path in work.baseline if is_test_file(path)}
    lock_note = None
    if protect_tests == "all":
        protected = all_protected
    else:
        protected = all_protected - test_files
        if protect_tests == "task":
            selected = judging_test_paths(tuple(test_files), test_command, ())
            if test_command and not selected:
                protected.update(test_files)
                lock_note = "selector-free task check: all existing test files locked"
            else:
                protected.update(selected)
            protected.update(declared)
    product = {path: text for path, text in texts.items()
               if path not in protected and (protect_tests != "all" or
                   not path.startswith(("tests/", "test/", "__tests__/")))}
    selected_target = repo_relative(target) if target else None
    if selected_target is not None and selected_target not in product:
        raise TaskRowError(f"--target {selected_target} is not a product source file in the repository")
    edits = [repo_relative(path) for path in edit_paths or []]
    bad = [path for path in edits if path not in product]
    if bad:
        raise TaskRowError("--edit file(s) are not product source files in the repository: "
                           + ", ".join(bad))
    scoped = bool(edits or len(product) > DECLARE_ALL_LIMIT)
    if scoped:
        scope = list(dict.fromkeys(([selected_target] if selected_target else []) + edits))
        if not scope:
            steering = set(declared) | set(judging_test_paths(tuple(test_files), test_command, ()))
            scope = auto_scope(task, product, {path: texts[path] for path in sorted(steering)
                                               if path in texts})
        if not scope:
            raise TaskRowError(f"the repository has {len(product)} source files; pass --target "
                               "(and --edit for any other file the fix may touch)")
        product = {path: product[path] for path in scope}
    source_files = {path: text for path, text in product.items() if path not in test_files}
    candidates = source_files or product
    entry = selected_target or next(
        (path for path in sorted(candidates)
         if Path(path).suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
                                  ".c", ".cc", ".cpp", ".sh"}
         and not path.endswith("/__init__.py")), None)
    entry = entry or guess_entry(candidates, {})
    if entry is None:
        entry = _output_hint(task, product)
    if entry is not None and not _public_hint(entry):
        entry = _output_hint(task, product)
        if entry is None:
            entry = guess_entry({path: text for path, text in product.items()
                                 if _public_hint(path)}, {})
    if entry is None:
        raise TaskRowError("could not pick a starting file; name an output path in --task")
    new_roots = [entry.split("/", 1)[0]] if entry not in product else []
    digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:12]
    identifier = task_id or f"task-{digest}"
    row = {
        "task_id": identifier, "sha": identifier, "repo": "build",
        "shape": "multi_file", "task_kind": "bug", "tier": "USER-REPO",
        "msg": task.strip(),
        "files": {path: {"parent_src": text} for path, text in sorted(product.items())},
        "scoped_edits": scoped,
        "public_tests": {},
        "protected_test_paths": sorted(protected),
        "runner_kind": "generic", "test_command": list(test_command),
        "allowed_product_paths": sorted({*(path.split("/", 1)[0] for path in product),
                                         *new_roots}),
        "new_file_roots": new_roots, "entry": {"fn": "", "path": entry},
        "path": entry, "rel": entry, "fn": "", "_skip_db": True,
    }
    if protect_tests != "all":
        row["protect_tests"] = protect_tests
        row["test_file_paths"] = sorted(test_files)
    if lock_note:
        row["test_lock_note"] = lock_note
    return row
