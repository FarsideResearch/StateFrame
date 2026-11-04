"""Prevent unrelated tests from sharing source-baseline environment variables."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


TESTS = Path(__file__).resolve().parent
RETIRED_VARIABLE = "SF_TEST_PARITY_BASELINE"


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _baseline_contract(source: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(source)
    os_modules = {"os"}
    environ_names: set[str] = set()
    getenv_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_modules.update(alias.asname or alias.name for alias in node.names if alias.name == "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "environ":
                    environ_names.add(alias.asname or alias.name)
                elif alias.name == "getenv":
                    getenv_names.add(alias.asname or alias.name)

    def is_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id in os_modules
        ) or (isinstance(node, ast.Name) and node.id in environ_names)

    environment_names: set[str] = set()
    pinned_commits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and is_environ(node.value):
            name = _literal_string(node.slice)
            if name is not None and "BASELINE" in name.upper():
                environment_names.add(name)
        elif isinstance(node, ast.Call):
            function = node.func
            environment_read = (
                isinstance(function, ast.Attribute)
                and function.attr == "get"
                and is_environ(function.value)
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "getenv"
                and isinstance(function.value, ast.Name)
                and function.value.id in os_modules
            ) or (isinstance(function, ast.Name) and function.id in getenv_names)
            if environment_read:
                name = _literal_string(node.args[0]) if node.args else None
                if name is None:
                    name = next(
                        (_literal_string(keyword.value) for keyword in node.keywords if keyword.arg == "key"),
                        None,
                    )
                if name is not None and "BASELINE" in name.upper():
                    environment_names.add(name)

            is_pinned_tree = (
                isinstance(function, ast.Name) and function.id == "pinned_tree"
            ) or (isinstance(function, ast.Attribute) and function.attr == "pinned_tree")
            if is_pinned_tree and node.args:
                commit = _literal_string(node.args[0])
                if commit is not None:
                    pinned_commits.add(commit)
    return environment_names, pinned_commits


def _baseline_violations(sources: dict[str, str]) -> list[str]:
    readers: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    violations = []
    for filename, source in sorted(sources.items()):
        environment_names, pinned_commits = _baseline_contract(source)
        if RETIRED_VARIABLE in environment_names:
            violations.append(f"{filename} reads retired baseline variable {RETIRED_VARIABLE}")
        for name in environment_names:
            readers[name].append((filename, pinned_commits))

    for name, entries in sorted(readers.items()):
        if len(entries) < 2:
            continue
        details = []
        for filename, commits in entries:
            pins = ", ".join(sorted(commits)) if commits else "no literal pinned_tree commit"
            details.append(f"{filename} pins {pins}")
        violations.append(f"baseline variable {name} is read by multiple test sources: " + "; ".join(details))
    return sorted(violations)


def test_test_sources_have_unique_baseline_variables():
    sources = {
        path.relative_to(TESTS).as_posix(): path.read_text(encoding="utf-8")
        for path in TESTS.rglob("test_*.py")
    }
    violations = _baseline_violations(sources)
    assert violations == [], "\n".join(violations)


def test_planted_duplicate_variable_reports_both_pins():
    sources = {
        "first/test_cache.py": (
            'import os\n'
            'os.environ.get("SF_TEST_SHARED_BASELINE")\n'
            'pinned_tree("e19c325", tmp_path_factory)\n'
        ),
        "second/test_scope.py": (
            'import os\n'
            'os.environ["SF_TEST_SHARED_BASELINE"]\n'
            'pinned_tree("0b057d3", tmp_path_factory)\n'
        ),
    }
    violations = _baseline_violations(sources)
    assert len(violations) == 1
    message = violations[0]
    for expected in (
        "first/test_cache.py",
        "e19c325",
        "second/test_scope.py",
        "0b057d3",
    ):
        assert expected in message


def test_planted_retired_variable_is_reported():
    violations = _baseline_violations({
        "test_legacy.py": 'import os\nos.getenv("SF_TEST_PARITY_BASELINE")\n'
    })
    assert violations == [
        "test_legacy.py reads retired baseline variable SF_TEST_PARITY_BASELINE"
    ]
