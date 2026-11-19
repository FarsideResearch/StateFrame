from __future__ import annotations

from pathlib import Path

import pytest

from stateframe.task_row import build_row
from stateframe.test_runners import discover_check, parse_command
from stateframe.workcopy import create


@pytest.mark.parametrize("resource", [
    "tests/README.md",
    "tests/helper.py",
    "tests/conftest.py",
])
def test_non_test_resources_do_not_hide_declared_python_build(tmp_path, resource):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='sample'\n")
    (repo / "app.py").write_text("value = 1\n")
    resource_path = repo / resource
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    resource_path.write_text("No tests implemented yet.\n")

    command, source, found = discover_check(repo)

    assert (command, source, found) == (
        "python -m compileall -q app.py", "build", False)
    runner_kind, argv = parse_command(command)
    build_row(create(repo, tmp_path / "work"), task="Update app.py",
              runner_kind=runner_kind, test_command=argv, protect_tests="off")
