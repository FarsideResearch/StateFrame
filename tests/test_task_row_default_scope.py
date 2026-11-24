from __future__ import annotations

import pytest

from stateframe.task_row import SCOPE_INFERRED, resolve_edit_scope


@pytest.mark.parametrize(("task", "check_count", "expected_path"), [
    ("Fix pricing", 0, None),
    ("Fix main.py", 0, "main.py"),
    ("Fix pkg/m0.py", 0, "pkg/m0.py"),
    ("Fix pkg/m0.py", 9, "pkg/m0.py"),
    ("Fix pricing", 9, None),
])
def test_large_repo_scope_is_narrow_or_enters_discovery(task, check_count, expected_path):
    product = {"main.py": "", **{f"pkg/m{index}.py": "" for index in range(200)}}
    checks = {f"tests/test_{index}.py": "" for index in range(check_count)}

    if expected_path is None:
        assert resolve_edit_scope(
            task, product, target=None, edits=[], steering=checks
        ) == ({}, True, "discovery")
        return

    files, scoped, source = resolve_edit_scope(
        task, product, target=None, edits=[], steering=checks)
    assert list(files) == [expected_path]
    assert scoped is True
    assert source == SCOPE_INFERRED
