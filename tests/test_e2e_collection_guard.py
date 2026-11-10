"""Keep top-level e2e tests collectible while fixture repositories stay data."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from _stateframe_home import isolated_stateframe_home


pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
E2E_TESTS = Path("tests/e2e")


def _e2e_python_sources(root: Path) -> list[str]:
    e2e_root = root / E2E_TESTS
    return [path.relative_to(root).as_posix() for path in e2e_root.rglob("*.py")]


def _fixture_repo_conftests(root: Path, python_sources: list[str] | None = None) -> list[str]:
    if python_sources is None:
        python_sources = _e2e_python_sources(root)
    return sorted(
        path
        for path in python_sources
        if PurePosixPath(path).name == "conftest.py" and PurePosixPath(path).parent != E2E_TESTS
    )


def _assert_e2e_collection(root: Path) -> tuple[str, ...]:
    """Collect the suite in a fresh process and enforce the e2e/fixture boundary."""
    python_sources = _e2e_python_sources(root)
    fixture_conftests = _fixture_repo_conftests(root, python_sources)
    if fixture_conftests:
        raise AssertionError(
            "fixture-repository conftest.py files found under tests/e2e: "
            + ", ".join(fixture_conftests)
        )

    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_HOME": isolated_stateframe_home(),
    }
    # Fix root-relative node IDs in synthetic projects while preserving full suite collection.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(root),
            "tests",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    node_ids = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    expected_files = sorted(
        path
        for path in python_sources
        if PurePosixPath(path).parent == E2E_TESTS and PurePosixPath(path).name.startswith("test_")
    )
    collected_files = {node_id.split("::", 1)[0] for node_id in node_ids}
    missing_files = [path for path in expected_files if path not in collected_files]
    nested_node_ids = [
        node_id
        for node_id in node_ids
        if node_id.split("::", 1)[0].startswith("tests/e2e/")
        and len(PurePosixPath(node_id.split("::", 1)[0]).parts) > 3
    ]
    output = result.stdout + result.stderr
    nested_sources = [path for path in python_sources if PurePosixPath(path).parent != E2E_TESTS]
    imported_nested_sources = [path for path in nested_sources if path in output]

    problems = []
    if missing_files:
        problems.append(f"top-level e2e test files without node ids: {missing_files}")
    if nested_node_ids:
        problems.append(f"fixture-repository node ids were collected: {nested_node_ids}")
    if imported_nested_sources:
        problems.append(f"fixture-repository Python sources appeared during collection: {imported_nested_sources}")
    if problems:
        summary = next(
            (line.strip() for line in reversed(output.splitlines()) if " collected" in line),
            "collection summary unavailable",
        )
        raise AssertionError(
            f"e2e collection contract failed (pytest exit {result.returncode}; {summary}): "
            + "; ".join(problems)
        )
    return node_ids


def _write_planted_tree(root: Path, ignore_glob: str) -> Path:
    tests = root / "tests"
    e2e = tests / "e2e"
    repo = e2e / "repo"
    repo.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        f"collect_ignore_glob = [{ignore_glob!r}]\n",
        encoding="utf-8",
    )
    (e2e / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    imported = repo / "imported.txt"
    (repo / "test_inner.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n"
        "def test_inner():\n"
        "    pass\n",
        encoding="utf-8",
    )
    return imported


def test_full_suite_collects_e2e_modules_and_ignores_fixture_repositories():
    _assert_e2e_collection(ROOT)


def test_collection_guard_detects_the_old_glob_and_accepts_the_fixed_glob(pytester):
    imported = _write_planted_tree(pytester.path, "e2e/*")

    with pytest.raises(AssertionError, match=r"tests/e2e/test_x\.py"):
        _assert_e2e_collection(pytester.path)
    assert not imported.exists()

    (pytester.path / "tests/conftest.py").write_text(
        'collect_ignore_glob = ["e2e/*/*"]\n',
        encoding="utf-8",
    )
    node_ids = _assert_e2e_collection(pytester.path)
    assert any(node_id.startswith("tests/e2e/test_x.py::") for node_id in node_ids)
    assert not any("tests/e2e/repo/" in node_id for node_id in node_ids)
    assert not imported.exists()

    conftest = pytester.path / "tests/e2e/repo/conftest.py"
    conftest.write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert _fixture_repo_conftests(pytester.path) == ["tests/e2e/repo/conftest.py"]
    with pytest.raises(AssertionError, match=r"tests/e2e/repo/conftest\.py"):
        _assert_e2e_collection(pytester.path)
    assert not imported.exists()

    conftest.unlink()
    assert _fixture_repo_conftests(pytester.path) == []
    _assert_e2e_collection(pytester.path)
    assert not imported.exists()
