"""RED tests for the task file-scope rule ("Change no other file", "Only edit X").

Imports live inside each test so every test is collected and fails on its own
until the target API exists.
"""
import pytest

TASK = ("Write your final answer to answer.json in the repository root: a JSON object "
        "with exactly two keys. Change no other file.")


def test_parse_task_file_scope_reads_answer_json_from_real_wording():
    from stateframe.task_file_scope import parse_task_file_scope
    assert parse_task_file_scope(TASK) == ("answer.json",)


def test_parse_task_file_scope_reads_only_edit_rule():
    from stateframe.task_file_scope import parse_task_file_scope
    assert parse_task_file_scope("Fix the bug in src/app.py. Only edit src/app.py.") == ("src/app.py",)


def test_parse_task_file_scope_reads_do_not_modify_rule_with_named_path():
    from stateframe.task_file_scope import parse_task_file_scope
    assert parse_task_file_scope(
        "Do not modify any other file; update config/settings.toml to set debug=false."
    ) == ("config/settings.toml",)


def test_parse_task_file_scope_none_without_scope_rule():
    from stateframe.task_file_scope import parse_task_file_scope
    assert parse_task_file_scope(
        "Fix the failing test in tests/test_x.py by changing src/x.py.") is None


def test_parse_task_file_scope_fails_open_when_rule_names_no_path():
    from stateframe.task_file_scope import parse_task_file_scope
    assert parse_task_file_scope("Change no other file.") is None


def test_resolve_edit_scope_task_rule_yields_only_existing_allowed_files():
    from stateframe.task_row import SCOPE_TASK_TEXT, resolve_edit_scope
    assert SCOPE_TASK_TEXT == "task_text"
    files, scoped, source = resolve_edit_scope(TASK, {"README.md": "x"}, target=None,
                                               edits=[], steering={})
    assert (files, scoped, source) == ({}, True, "task_text")
    product = {"README.md": "x", "answer.json": "{}"}
    files, scoped, source = resolve_edit_scope(TASK, product, target=None, edits=[], steering={})
    assert (files, scoped, source) == ({"answer.json": "{}"}, True, "task_text")


def test_resolve_edit_scope_without_task_rule_is_not_task_text_control():
    from stateframe.task_row import resolve_edit_scope
    _, _, source = resolve_edit_scope("Fix README.md", {"README.md": "x"}, target=None,
                                      edits=[], steering={})
    assert source != "task_text"


def test_is_product_path_excludes_paths_outside_task_file_scope():
    from probes.react_repair import _is_product_path
    # The slice shape: an ordinary user repo whose task rule scoped the run.
    row = {"tier": "USER-REPO", "scoped_edits": True, "runner_kind": "generic",
           "task_file_scope": ["answer.json"]}
    assert _is_product_path("answer.json", row) is True  # a named new file stays creatable
    assert _is_product_path("notes.md", row) is False
    assert _is_product_path("tests/answer.json", row) is False


def test_is_product_path_unscoped_user_repo_unchanged():
    """Control: without a task rule an unscoped user repo still admits new files."""
    from probes.react_repair import _is_product_path
    row = {"tier": "USER-REPO", "scoped_edits": False, "runner_kind": "generic"}
    assert _is_product_path("notes.md", row) is True


def test_workspace_policy_refuses_paths_outside_task_scope():
    from kernel_effects import Effect, Entry, WorkspaceEffectPolicy
    policy = WorkspaceEffectPolicy(protected=lambda p: False, edit_scope=lambda p: True,
                                   new_file_allowed=lambda p: True, task_scope=("answer.json",))
    entry = Entry("file", 0o644, "a" * 64, 1)
    assert policy.authorize(Effect("README.md", "modified", entry, entry)) == (
        False, "outside task file scope")
    assert policy.authorize(Effect("answer.json", "modified", entry, entry))[0] is True


def test_out_of_task_scope_paths_returns_sorted_candidates_not_in_scope():
    from stateframe.runner import _out_of_task_scope_paths
    candidate = {"answer.json": b"1", "README.md": b"x", "gone.txt": None}
    assert _out_of_task_scope_paths(candidate, ["answer.json"]) == ["README.md", "gone.txt"]
