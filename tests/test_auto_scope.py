"""Large repositories infer a narrow edit scope or require explicit file flags."""
from pathlib import Path

import pytest

from stateframe import task_row
from stateframe.task_row import (
    AUTO_SCOPE_LIMIT,
    DECLARE_ALL_LIMIT,
    SCOPE_INFERRED,
    auto_scope,
    build_row,
)
from stateframe.workcopy import Limits, create

ROOT = Path(__file__).resolve().parents[1]
TEST_SRC = "from app.pricing import total\n\ndef test_total():\n    assert total(80, 25) == 60\n"


def _large_repo(tmp_path, *, extra_tests=0):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    for index in range(DECLARE_ALL_LIMIT + 5):
        (repo / "pkg" / f"mod_{index:03d}.py").write_text(f"VALUE = {index}\n")
    (repo / "app").mkdir()
    (repo / "app" / "__init__.py").write_text("")
    (repo / "app" / "pricing.py").write_text("from app.rates import rate\n\ndef total(a, p):\n    return a * rate(p)\n")
    (repo / "app" / "rates.py").write_text("from pkg.mod_007 import VALUE\n\ndef rate(p):\n    return 1 - p / 100\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_pricing.py").write_text(TEST_SRC)
    for index in range(extra_tests):
        (repo / "tests" / f"test_extra_{index}.py").write_text(f"import pkg.mod_{index:03d}\n")
    return create(repo, tmp_path / "copy")


def test_task_named_file_scopes_an_over_limit_repo(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Fix the discount math in app/pricing.py.")
    assert row["scoped_edits"] is True
    assert list(row["files"]) == ["app/pricing.py"]
    assert row["entry"]["path"] == "app/pricing.py"


def test_chosen_check_imports_scope_transitively(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Discounts are charged wrong.", checks=["tests/test_pricing.py"])
    # The check imports app.pricing, which imports app.rates, which imports pkg.mod_007.
    assert sorted(row["files"]) == ["app/pricing.py", "app/rates.py"]
    assert row["entry"]["path"] == "app/pricing.py"


def test_test_command_operand_steers_the_scope(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Discounts are charged wrong.", runner_kind="generic",
                    test_command=("python", "-m", "pytest", "tests/test_pricing.py"))
    assert sorted(row["files"]) == ["app/pricing.py", "app/rates.py"]


def test_nothing_to_go_on_starts_read_only_discovery(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Discounts are charged wrong.")
    assert row["scope_source"] == "discovery"
    assert row["scoped_edits"] is True and row["files"] == {}


def test_broad_check_selection_does_not_steer(tmp_path):
    work = _large_repo(tmp_path, extra_tests=task_row.AUTO_SCOPE_CHECK_LIMIT)
    row = build_row(work, task="Discounts are charged wrong.", runner_kind="generic",
                    test_command=("python", "-m", "pytest", "tests/"))
    assert row["scope_source"] == "discovery"
    assert row["scoped_edits"] is True and row["files"] == {}


def test_named_task_path_still_scopes_broad_check_selection(tmp_path):
    work = _large_repo(tmp_path, extra_tests=task_row.AUTO_SCOPE_CHECK_LIMIT)
    row = build_row(work, task="Fix pkg/mod_000.py.", runner_kind="generic",
                    test_command=("python", "-m", "pytest", "tests/"))
    assert row["scope_source"] == SCOPE_INFERRED
    assert list(row["files"]) == ["pkg/mod_000.py"]


def test_explicit_flags_win_over_the_automatic_scope(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Fix app/pricing.py.", target="app/rates.py",
                    checks=["tests/test_pricing.py"])
    assert list(row["files"]) == ["app/rates.py"]


def test_scope_is_capped_and_ordered():
    product = {f"lib/m{index:02d}.py": "" for index in range(AUTO_SCOPE_LIMIT + 6)}
    task = " ".join(sorted(product, reverse=True))
    chosen = auto_scope(task, product, {})
    assert len(chosen) == AUTO_SCOPE_LIMIT
    assert chosen == sorted(product, reverse=True)[:AUTO_SCOPE_LIMIT]


def test_named_paths_outside_the_product_are_ignored():
    product = {"app/pricing.py": ""}
    assert auto_scope("See tests/test_pricing.py and docs/x.md, then app/pricing.py.", product, {}) == [
        "app/pricing.py"]


@pytest.mark.parametrize("example,task", [
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
])
def test_rows_under_the_limit_are_unchanged(tmp_path, monkeypatch, example, task):
    work = create(ROOT / "examples" / example, tmp_path / "copy")
    before = build_row(work, task=task, checks=None)
    monkeypatch.setattr(task_row, "auto_scope", lambda *args: pytest.fail("auto scope used under the limit"))
    assert build_row(work, task=task, checks=None) == before
    assert before["scoped_edits"] is False


def test_named_file_is_recorded_as_inferred_scope(tmp_path):
    work = _large_repo(tmp_path)
    row = build_row(work, task="Fix the discount math in app/pricing.py.")
    assert row["scope_source"] == SCOPE_INFERRED


def test_6800_file_repo_with_unnamed_task_starts_read_only_discovery(tmp_path):
    """Build a real large row with an empty edit scope when neither task nor check locates the work.

    Built on disk (not mocked) so the real copy, text scan and row assembly run end to end.
    """
    repo = tmp_path / "repo"
    for index in range(6800):
        package = repo / f"pkg{index % 40:02d}"
        package.mkdir(parents=True, exist_ok=True)
        (package / f"mod_{index:05d}.py").write_text(f"VALUE = {index}\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_values.py").write_text("def test_values():\n    assert True\n")
    # The copy step has its own file ceiling (Limits.max_files=5000); lift it here so the test
    # exercises the task-row path rather than the copy refusal, which is a separate limit.
    work = create(repo, tmp_path / "copy", Limits(max_files=7000))
    row = build_row(work, task="Make the numbers add up.")
    assert row["scope_source"] == "discovery"
    assert row["scoped_edits"] is True and row["files"] == {}
