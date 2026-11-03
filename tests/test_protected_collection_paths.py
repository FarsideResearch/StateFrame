"""The collection guard accepts baseline test paths in both forms the engine passes.

Engine callers hand effect_policy_for_run absolute paths under the workspace; a projected worker row carries
workspace-relative ones (``tests/test_app.py``). The 0.3.26 guard called ``relative_to`` on every entry and
raised ``ValueError`` for the relative form, breaking effect authorization for projected rows.
"""
from pathlib import Path

from kernel_effects import _workspace_relative, effect_policy_for_run, scan


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("def test_ok(): assert True\n")
    return root


def test_relative_path_is_kept_as_is(tmp_path):
    assert _workspace_relative(tmp_path, "tests/test_app.py") == "tests/test_app.py"


def test_absolute_path_under_the_workspace_is_made_relative(tmp_path):
    assert _workspace_relative(tmp_path, tmp_path / "tests/test_app.py") == "tests/test_app.py"


def test_absolute_path_through_a_symlinked_root_is_made_relative(tmp_path):
    real = tmp_path / "real"
    (real / "tests").mkdir(parents=True)
    (real / "tests/test_app.py").write_text("")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert _workspace_relative(link, real / "tests/test_app.py") == "tests/test_app.py"


def _row():
    return {"files": {"app.py": "VALUE = 1\n"}, "runner_kind": "pytest", "protect_tests": "all",
            "test_file_paths": ["tests/test_app.py"]}


def test_projected_row_with_relative_baseline_builds_a_policy(tmp_path):
    root = _workspace(tmp_path)
    before = scan(root, exclude=lambda _: False)
    policy = effect_policy_for_run(root, _row(), {"tests/test_app.py"}, baseline_manifest=before)
    assert policy is not None


def test_relative_and_absolute_baselines_give_the_same_policy_inputs(tmp_path):
    root = _workspace(tmp_path)
    before = scan(root, exclude=lambda _: False)
    relative = effect_policy_for_run(root, _row(), {"tests/test_app.py"}, baseline_manifest=before)
    absolute = effect_policy_for_run(root, _row(), {str(root / "tests/test_app.py")}, baseline_manifest=before)
    assert type(relative) is type(absolute)
