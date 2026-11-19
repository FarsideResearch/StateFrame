"""Regression coverage for affirmative, repository-relative task scope rules."""

import pytest

from stateframe.task_file_scope import TaskFileScopeError, parse_task_file_scope


@pytest.mark.parametrize(("task", "expected"), [
    # Review finding 3: forbidden paths, examples and URL metadata.
    ("Update answer.json. Do not modify config.py. Change no other file.",
     ("answer.json",)),
    ("Only edit a.py and not b.py.", ("a.py",)),
    ('Only edit a.py. Example: "Change no other file; update b.py."', ("a.py",)),
    ("Only edit a.py using https://example.com/?file=b.py.", ("a.py",)),
    ("Only edit a.py using https://example.com/#b.py.", ("a.py",)),
    # Review finding 5: conditional/documentary rules and dotted prose.
    ("Fix src/app.py. Only edit tests/test_app.py if needed.", None),
    ('Update README.md to document the example "Only edit src/app.py."', None),
    ("Only edit the implementation, i.e. src/app.py.", ("src/app.py",)),
    ("Only edit the implementation, e.g. src/app.py.", ("src/app.py",)),
    ("Only edit src/app.py for version v1.alpha.", ("src/app.py",)),
    # Review finding 6: complete path tokens and backward references.
    ("Only edit ./src/app.py.", ("src/app.py",)),
    ("Only edit `./src/app.py`.", ("src/app.py",)),
    (r"Only edit src\app.py.", ("src/app.py",)),
    (r"Only edit C:\repo\src\app.py.", TaskFileScopeError),
    ("src/app.py is the file to repair. Only edit the file above.", ("src/app.py",)),
    ("Only edit Dockerfile.", ("Dockerfile",)),
    ("Only edit .github/workflows/ci.yml.", (".github/workflows/ci.yml",)),
    # Full output instruction, including quoted JSON keys.
    (('Write your final answer to answer.json in the repository root: a JSON object '
      'with exactly two keys, "answer" ... and "evidence_ids" ... '
      'Cite each effective receipt. Change no other file.'), ("answer.json",)),
    ("Fix src/app.py and update tests/test_app.py.", None),
])
def test_review_regressions(task, expected):
    if expected is TaskFileScopeError:
        with pytest.raises(TaskFileScopeError):
            parse_task_file_scope(task)
    else:
        assert parse_task_file_scope(task) == expected


@pytest.mark.parametrize("task", [
    "Answer only by creating the file ANSWER.json at the repository root. "
    "Do not change any other file.",
    "Answer only by creating the file ANSWER.json. Do not change any other file.",
    "Create ANSWER.json at the repository root. Do not change any other file.",
    "Create ANSWER.json. Do not change any other file.",
    "Write the file ANSWER.json at the repository root. Do not change any other file.",
    "Write the file ANSWER.json. Do not change any other file.",
    "Add the file ANSWER.json at the repository root. Do not change any other file.",
    "Add the file ANSWER.json. Do not change any other file.",
])
def test_answer_output_file_wording_parses_in_strict_mode(task):
    assert parse_task_file_scope(task, strict=True) == ("ANSWER.json",)


@pytest.mark.parametrize("verb", ["edit", "change", "modify", "write"])
def test_only_rule_consumes_affirmative_target_list(verb):
    assert parse_task_file_scope(
        f"Only {verb} a.py, `./src/b.py` and .github/workflows/ci.yml."
    ) == ("a.py", "src/b.py", ".github/workflows/ci.yml")


@pytest.mark.parametrize("suffix", [
    "and not b.py", "not b.py", "except b.py", "for b.py", "using b.py",
    "to fix b.py", "and document b.py", "i.e. b.py", "e.g. b.py",
])
def test_only_rule_stops_after_governed_targets(suffix):
    assert parse_task_file_scope(f"Only edit a.py {suffix}.") == ("a.py",)


@pytest.mark.parametrize("abbreviation", ["i.e.", "e.g.", "I.E.", "E.G."])
def test_abbreviation_can_introduce_target(abbreviation):
    assert parse_task_file_scope(f"Only edit {abbreviation} src/app.py.") == ("src/app.py",)
    with pytest.raises(TaskFileScopeError):
        parse_task_file_scope(f"Only edit {abbreviation}")


@pytest.mark.parametrize("quoted", [
    '"Only edit b.py."', "'Only edit b.py.'", "`Only edit b.py.`",
    '"Change no other file; update b.py."',
    "```text\nOnly edit b.py.\n```", "~~~text\nOnly edit b.py.\n~~~",
])
def test_quoted_rules_cannot_add_targets_or_trigger_rules(quoted):
    assert parse_task_file_scope(f"Only edit a.py. Example: {quoted}") == ("a.py",)
    assert parse_task_file_scope(f"Update README.md to document {quoted}") is None


@pytest.mark.parametrize("url", [
    "https://example.com/?file=b.py#c.py", "https://example.com/#Only-edit-b.py",
    "https://example.com/?q=Only%20edit%20b.py", "file:///tmp/b.py",
])
def test_url_paths_are_not_affirmative_targets(url):
    assert parse_task_file_scope(
        f"Update a.py using {url}. Change no other file."
    ) == ("a.py",)


@pytest.mark.parametrize("verb", [
    "write", "create", "edit", "change", "modify", "update", "save", "put", "fix", "add",
])
def test_no_other_rule_collects_affirmative_write_targets(verb):
    assert parse_task_file_scope(
        f"{verb.capitalize()} a.py and src/b.py. Change no other file."
    ) == ("a.py", "src/b.py")


@pytest.mark.parametrize("negative", [
    "Do not modify", "Don't edit", "Never update", "Do not create", "Do not change",
])
def test_negated_clauses_do_not_grant_permissions(negative):
    assert parse_task_file_scope(
        f"Update answer.json and {negative.lower()} config.py. Change no other file."
    ) == ("answer.json",)
    assert parse_task_file_scope(f"{negative} config.py. Change no other file.") is None


@pytest.mark.parametrize("name", ["Dockerfile", "Makefile", "LICENSE", "README"])
def test_extensionless_targets_are_governed_by_write_verbs(name):
    assert parse_task_file_scope(f"Only edit {name}.") == (name,)
    assert parse_task_file_scope(f"Update {name}. Do not change any other file.") == (name,)
    assert parse_task_file_scope(f"Read {name}. Change no other file.") is None


@pytest.mark.parametrize("path", [
    "/etc/config.py", r"C:\repo\app.py", "C:/repo/app.py", r"\\server\share\app.py",
    "../app.py", "src/../../app.py", "~/app.py",
])
def test_unsupported_target_never_becomes_a_basename(path):
    with pytest.raises(TaskFileScopeError):
        parse_task_file_scope(f"Only edit {path}.")
    with pytest.raises(TaskFileScopeError):
        parse_task_file_scope(f"Only edit a.py and {path}.")
    with pytest.raises(TaskFileScopeError):
        parse_task_file_scope(f"Update {path}. Change no other file.")


@pytest.mark.parametrize("task", [
    "Only edit the file above.",
    "src/a.py and src/b.py need repairs. Only edit the file above.",
    "src/a.py needs repairs. No file is named here. Only edit the file above.",
    "Only edit v1.alpha.",
    "Fix a.py. If needed, only edit b.py.",
])
def test_ambiguous_scope_degrades_for_row_callers(task, caplog):
    import logging

    from stateframe.task_row import resolve_edit_scope

    caplog.set_level(logging.WARNING, logger="stateframe.task_row")
    product = {"a.py": "A = 1\n", "src/a.py": "A = 1\n", "src/b.py": "B = 2\n"}
    assert resolve_edit_scope(task, product, target=None, edits=[], steering={}) == (
        product, False, "whole_repo")
    events = [record for record in caplog.records
              if getattr(record, "event_type", None) == "task_file_scope_degraded"]
    assert len(events) == 1


@pytest.mark.parametrize("task", [
    "Only edit tests if needed.", "Change no other file.",
    "Fix a.py. If needed, only edit b.py.",
    "Fix a.py; only edit b.py if necessary.",
])
def test_ambiguous_or_conditional_rule_fails_open(task):
    assert parse_task_file_scope(task) is None


def test_unique_previous_path_is_normalized_and_deduplicated():
    assert parse_task_file_scope(
        "`./src/app.py` replaces src/app.py. Only edit the file above."
    ) == ("src/app.py",)


def test_repeated_targets_preserve_first_mention_order():
    assert parse_task_file_scope(
        "Only edit b.py, a.py and b.py. Only modify ./a.py."
    ) == ("b.py", "a.py")


@pytest.mark.parametrize("task", [None, 42, "", " \n\t"])
def test_missing_task_fails_open(task):
    assert parse_task_file_scope(task) is None


@pytest.mark.parametrize("negative", [
    "Do not ever modify config.py", "Never directly edit config.py",
    "Do not modify config.py or update secret.py",
    "Don't edit config.py and change secret.py",
])
def test_negation_covers_adverbs_and_coordinated_verbs(negative):
    assert parse_task_file_scope(
        f"Update answer.json. {negative}. Change no other file."
    ) == ("answer.json",)


@pytest.mark.parametrize("prefix", ["/etc/example:", "~/example:", r"C:\example:"])
def test_quoted_rule_is_masked_even_after_unsupported_path_prefix(prefix):
    assert parse_task_file_scope(
        f'Only edit a.py. Example: "{prefix} Only edit b.py."'
    ) == ("a.py",)


@pytest.mark.parametrize("continuation", [
    "and only edit answer.json", "but update answer.json", "then write answer.json",
])
def test_new_affirmative_directive_after_negation(continuation):
    assert parse_task_file_scope(
        f"Do not modify config.py {continuation}. Change no other file."
    ) == ("answer.json",)
