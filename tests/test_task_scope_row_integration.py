"""Task file limits survive row construction and product snapshot admission."""

from pathlib import Path

import pytest

from probes.react_repair import FileSet, _is_product_path
from stateframe.task_row import DECLARE_ALL_LIMIT, TaskRowError, build_row
from stateframe.workcopy import WorkCopy, create


ANSWER_TASK = (
    "Write your final answer to answer.json in the repository root: "
    "a JSON object with exactly two keys. Change no other file."
)
CHECK = "tests/test_check.py"
EDITABLE_TEST = "tests/test_task_file_scope_red.py"


@pytest.fixture(params=("generic", "pytest"))
def runner_kind(request):
    return request.param


def make_work(tmp_path: Path, runner_kind: str, files: dict[str, str]) -> WorkCopy:
    repo = tmp_path / "repo"
    sources = dict(files)
    if runner_kind == "pytest":
        sources.setdefault(CHECK, "def test_check():\n    assert True\n")
    for relative, content in sources.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return create(repo, tmp_path / "copy")


@pytest.mark.parametrize(("task", "output"), (
    (ANSWER_TASK, "answer.json"),
    ("Only edit new.py.", "new.py"),
    ("Only edit output/new.py.", "output/new.py"),
))
def test_named_new_file_is_the_starting_entry(tmp_path, runner_kind, task, output):
    work = make_work(tmp_path, runner_kind, {"README.md": "Repository\n"})
    row = build_row(work, task=task, runner_kind=runner_kind, test_command=("true",))

    assert row["files"] == {}
    assert row["task_file_scope"] == [output]
    assert row["scope_source"] == "task_text"
    assert row["scoped_edits"] is True
    assert row["entry"] == {"fn": "", "path": output}
    assert row["path"] == row["rel"] == output
    assert row["new_file_roots"] == [output.split("/", 1)[0]]
    assert row["allowed_product_paths"] == row["new_file_roots"]
    assert output not in work.baseline
    fileset = FileSet.from_row(row)
    assert fileset.primary_path == output
    assert fileset.parent == {}
    assert _is_product_path(output, row)
    assert not _is_product_path("notes.py", row)


def test_existing_named_file_is_the_only_source(tmp_path, runner_kind):
    work = make_work(tmp_path, runner_kind, {"README.md": "Repository\n", "answer.json": "{}\n"})
    row = build_row(work, task=ANSWER_TASK, runner_kind=runner_kind, test_command=("true",))

    assert row["files"] == {"answer.json": {"parent_src": "{}\n"}}
    assert row["task_file_scope"] == ["answer.json"]
    assert row["entry"]["path"] == "answer.json"
    assert row["new_file_roots"] == []


def test_no_rule_keeps_small_repository_unscoped(tmp_path, runner_kind):
    sources = {"README.md": "Repository\n", "a.py": "A = 1\n", "b.py": "B = 2\n"}
    work = make_work(tmp_path, runner_kind, sources)
    row = build_row(work, task="Fix a.py.", runner_kind=runner_kind, test_command=("true",))

    assert row["files"] == {path: {"parent_src": content} for path, content in sources.items()}
    assert row["scope_source"] == "whole_repo"
    assert row["scoped_edits"] is False
    assert "task_file_scope" not in row


@pytest.mark.parametrize("selection", ("edit", "edit_with_target", "large_target"))
def test_conflicting_explicit_scope_is_actionable(tmp_path, runner_kind, selection):
    sources = {"a.py": "A = 1\n", "b.py": "B = 2\n"}
    if selection == "large_target":
        sources.update({f"extra_{index}.py": "X = 0\n" for index in range(DECLARE_ALL_LIMIT)})
    work = make_work(tmp_path, runner_kind, sources)
    with pytest.raises(TaskRowError) as error:
        build_row(
            work, task="Only edit a.py.", runner_kind=runner_kind, test_command=("true",),
            edit_paths=["b.py"] if selection == "edit" else
            ["a.py"] if selection == "edit_with_target" else None,
            target="b.py" if selection != "edit" else None,
        )

    message = str(error.value)
    assert "a.py" in message and "b.py" in message
    assert "task" in message.lower()
    assert "--edit" in message or "--target" in message


@pytest.mark.parametrize("selection", ("edit", "edit_with_target", "large_target"))
@pytest.mark.parametrize("task", ("Only edit a.py.", "Only edit a.py and b.py."))
def test_matching_explicit_scope_retains_task_cap(tmp_path, runner_kind, selection, task):
    sources = {"a.py": "A = 1\n", "b.py": "B = 2\n"}
    if selection == "large_target":
        sources.update({f"extra_{index}.py": "X = 0\n" for index in range(DECLARE_ALL_LIMIT)})
    work = make_work(tmp_path, runner_kind, sources)
    row = build_row(
        work, task=task, runner_kind=runner_kind, test_command=("true",),
        edit_paths=["a.py"] if selection != "large_target" else None,
        target="a.py" if selection != "edit" else None,
    )

    assert row["files"] == {"a.py": {"parent_src": "A = 1\n"}}
    assert row["scope_source"] == "explicit"
    assert row["scoped_edits"] is True
    assert row["task_file_scope"] == (["a.py", "b.py"] if "b.py" in task else ["a.py"])
    assert not _is_product_path("notes.py", row)


def test_small_repository_target_remains_a_hint_with_task_scope(tmp_path, runner_kind):
    work = make_work(tmp_path, runner_kind, {"a.py": "A = 1\n", "b.py": "B = 2\n"})
    row = build_row(work, task="Only edit a.py.", target="b.py", runner_kind=runner_kind,
                    test_command=("true",))

    assert row["files"] == {"a.py": {"parent_src": "A = 1\n"}}
    assert row["entry"]["path"] == "a.py"
    assert row["task_file_scope"] == ["a.py"]
    assert row["scope_source"] == "task_text"
    assert FileSet.from_row(row).primary_path == "a.py"


def test_task_scope_cannot_turn_a_protected_check_into_a_new_file(tmp_path, runner_kind):
    work = make_work(tmp_path, runner_kind, {"README.md": "Repository\n", "judge.py": "pass\n"})
    with pytest.raises(TaskRowError):
        build_row(work, task="Only edit judge.py.", checks=["judge.py"],
                  runner_kind=runner_kind, test_command=("true",))


def test_generic_task_scope_keeps_protected_globs(tmp_path):
    work = make_work(tmp_path, "generic", {"README.md": "Repository\n", "output/result.json": "{}"})
    with pytest.raises(TaskRowError):
        build_row(work, task="Only edit output/result.json.", runner_kind="generic",
                  protected_test_globs=("output/**",), test_command=("true",))


@pytest.mark.parametrize(("task", "files"), (
    (r"Only edit C:\repo\src\app.py.", {"README.md": "Repository\n"}),
    ("src/a.py and src/b.py need repairs. Only edit the file above.", {
        "src/a.py": "A = 1\n", "src/b.py": "B = 2\n",
    }),
))
def test_unresolvable_task_scope_refuses_row(tmp_path, runner_kind, task, files):
    work = make_work(tmp_path, runner_kind, files)
    with pytest.raises(TaskRowError, match="task file-scope rule"):
        build_row(work, task=task, runner_kind=runner_kind, test_command=("true",))


@pytest.mark.parametrize("task", (
    "Change no other file.",
    "Do not modify config.py. Change no other file.",
    "Read Dockerfile. Change no other file.",
))
def test_pathless_no_other_rule_refuses_unscoped_row(tmp_path, runner_kind, task):
    work = make_work(tmp_path, runner_kind, {"README.md": "Repository\n", "config.py": "x = 1\n"})
    with pytest.raises(TaskRowError, match="task file-scope rule"):
        build_row(work, task=task, runner_kind=runner_kind, test_command=("true",))


@pytest.mark.parametrize("protection", ("task", "off"))
@pytest.mark.parametrize("allowed", (False, True))
def test_editable_tests_respect_task_scope(tmp_path, runner_kind, protection, allowed):
    work = make_work(tmp_path, runner_kind, {
        "a.py": "A = 1\n", CHECK: "def test_check(): pass\n",
        EDITABLE_TEST: "def test_editable(): pass\n",
    })
    task = f"Only edit a.py and {EDITABLE_TEST}." if allowed else "Only edit a.py."
    row = build_row(work, task=task, runner_kind=runner_kind, checks=[CHECK],
                    test_command=("python", "-m", "pytest", CHECK), protect_tests=protection)

    assert EDITABLE_TEST in row["test_file_paths"]
    assert EDITABLE_TEST not in row["protected_test_paths"]
    assert _is_product_path(EDITABLE_TEST, row) is allowed
    assert _is_product_path("a.py", row)
    # Neither a named judging check nor an undeclared test-like create is product source.
    guarded = {**row, "task_file_scope": [CHECK, "tests/test_new.py", "test_new.py"],
               "protected_test_paths": [CHECK]}
    for path in guarded["task_file_scope"]:
        assert not _is_product_path(path, guarded)


@pytest.mark.parametrize("path", ("scratch/new.py", "private/new.py", "probes/new.py",
                                  "tests/new.py", "output/cache.pyc"))
def test_task_scope_does_not_remove_nonproduct_exclusions(path):
    row = {"tier": "USER-REPO", "scoped_edits": True, "runner_kind": "generic",
           "task_file_scope": [path], "new_file_roots": [path.split("/", 1)[0]]}
    assert not _is_product_path(path, row)
