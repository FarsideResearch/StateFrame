"""Regression checks for mutation path grants at the host boundary."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("grant", "unintended"), [
    ("app/[slug]/page.tsx", "app/s/page.tsx"),
    ("[x].csv", "x.csv"),
    ("data/*.csv", "data/private/payroll.csv"),
    ("src/*.py", "src/admin/access.py"),
])
def test_mutation_scope_grants_do_not_admit_reviewed_unintended_reads(
        tmp_path: Path, grant: str, unintended: str) -> None:
    from kernel_mutate_inputs import input_path_admitted
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    target = repo / unintended
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("private\n")
    policy = MutationPolicy(level="authored", read_globs=(grant,))
    assert not input_path_admitted(repo, unintended, policy)
    assert not input_path_admitted(
        repo, unintended, MutationPolicy(level="authored"), (grant,))


@pytest.mark.parametrize("literal", ["app/[slug]/page.tsx", "[x].csv"])
def test_mutation_scope_grants_admit_reviewed_literal_reads(
        tmp_path: Path, literal: str) -> None:
    from kernel_mutate_inputs import input_path_admitted
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    target = repo / literal
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("literal\n")
    assert input_path_admitted(
        repo, literal, MutationPolicy(level="authored", read_globs=(literal,)),
        (literal,))


@pytest.mark.parametrize(("grant", "unintended"), [
    ("app/[slug]/page.tsx", "app/s/page.tsx"),
    ("[x].csv", "x.csv"),
    ("data/*.csv", "data/private/payroll.csv"),
    ("src/*.py", "src/admin/access.py"),
])
def test_mutation_scope_grants_do_not_admit_reviewed_unintended_edits(
        tmp_path: Path, grant: str, unintended: str) -> None:
    from kernel_mutate_edits import EditConflict, MutationEdits
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    (repo / unintended).parent.mkdir(parents=True, exist_ok=True)
    runtime = SimpleNamespace(
        repo=repo, policy=MutationPolicy(level="authored", write_globs=(grant,)),
        effect_row=None, edit_scope=None, protected_paths=set())
    with pytest.raises(EditConflict, match="write grant"):
        MutationEdits(runtime).stage("scope-test", [
            {"op": "create", "path": unintended, "text": "new\n"}])


@pytest.mark.parametrize("literal", ["app/[slug]/page.tsx", "[x].csv"])
def test_mutation_scope_grants_admit_reviewed_literal_edits(
        tmp_path: Path, literal: str) -> None:
    from kernel_mutate_edits import MutationEdits
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    (repo / literal).parent.mkdir(parents=True, exist_ok=True)
    runtime = SimpleNamespace(
        repo=repo, policy=MutationPolicy(level="authored", write_globs=(literal,)),
        effect_row=None, edit_scope=None, protected_paths=set())
    staged = MutationEdits(runtime).stage("scope-test", [
        {"op": "create", "path": literal, "text": "new\n"}])
    assert staged.files[0]["path"] == literal


def test_mutation_scope_recursive_double_star_is_the_only_directory_wildcard(
        tmp_path: Path) -> None:
    from kernel_mutate_inputs import input_path_admitted
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    for relative in ("data/scores.csv", "data/private/payroll.csv"):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("csv\n")
    policy = MutationPolicy(level="authored", read_globs=("data/**",))
    assert input_path_admitted(repo, "data/scores.csv", policy)
    assert input_path_admitted(repo, "data/private/payroll.csv", policy)


def test_mutation_scope_recursive_grant_excludes_its_parent_file(
        tmp_path: Path) -> None:
    from kernel_mutate_inputs import input_path_admitted
    from kernel_mutate_policy import MutationPolicy

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").write_text("parent file\n")
    assert not input_path_admitted(
        repo, "data", MutationPolicy(level="authored", read_globs=("data/**",)))


@pytest.mark.parametrize(("path", "grant", "expected"), [
    ("app/[slug]/a.tsx", "app/[slug]/?.tsx", True),
    ("app/s/a.tsx", "app/[slug]/?.tsx", False),
    ("app/[slug]/nested/a.tsx", "app/[slug]/?.tsx", False),
    ("src/admin/access.py", "src/**.py", False),
    ("src/admin/access.py", "src/**/*.py", True),
])
def test_mutation_scope_wildcards_keep_segment_boundaries_and_literal_brackets(
        path: str, grant: str, expected: bool) -> None:
    from kernel_mutate_inputs import scope_path_matches

    assert scope_path_matches(path, grant) is expected
