"""RED coverage for the closed gerund forms of explicit file-scope rules."""
from __future__ import annotations

import re

import pytest

from stateframe.task_file_scope import TaskFileScopeError, parse_task_file_scope


GERUNDS = (
    "editing",
    "changing",
    "modifying",
    "writing",
    "updating",
    "touching",
)

_T1_PREFIXES = (
    "",
    "By ",
    "While ",
    "Fix the bug by ",
    "Fix the bug while ",
)
_T2_PREFIXES = (
    "Only ",
    "Fix the bug by only ",
)
_T2_ENDINGS = (
    "",
    " is allowed",
    " is permitted",
)

_TEMPLATE_CASES = [
    (f"T1 {prefix or 'sentence start'} / {gerund}",
     f"{prefix}{gerund} only src/app.py.", ("src/app.py",))
    for gerund in GERUNDS
    for prefix in _T1_PREFIXES
] + [
    (f"T2 {prefix.strip() or 'sentence start'} / {gerund} / {ending or 'no suffix'}",
     f"{prefix}{gerund} src/app.py{ending}.", ("src/app.py",))
    for gerund in GERUNDS
    for prefix in _T2_PREFIXES
    for ending in _T2_ENDINGS
]

_EXAMPLE_CASES = [
    ("specified by-editing example", "Fix the bug by editing only src/app.py.",
     ("src/app.py",)),
    ("specified only-editing example", "Only editing src/app.py is allowed.",
     ("src/app.py",)),
    ("specified writing example", "Writing only src/a.py, fix the bug.",
     ("src/a.py",)),
    ("auxiliary words inside paths do not block a valid rule",
     "Review src/not.py and src/will.py by editing only src/app.py.",
     ("src/app.py",)),
]

POSITIVE_CASES = _TEMPLATE_CASES + _EXAMPLE_CASES

_PAST_FORMS = {
    "editing": "edited",
    "changing": "changed",
    "modifying": "modified",
    "writing": "written",
    "updating": "updated",
    "touching": "touched",
}

_NEGATIVE_PATTERNS = (
    "{gerund} files only src/app.py.",
    "{gerund} only carefully src/app.py.",
    "{gerund} only src/app.py carefully.",
    "{gerund} only src/app.py is allowed.",
    "By {gerund} only src/app.py then update src/other.py.",
    "While {gerund} only src/app.py and then update src/other.py.",
    "Could you fix the bug by {gerund} only src/app.py.",
    "Do not edit config.py while {gerund} only src/app.py.",
    "Only {gerund} src/app.py is eventually allowed.",
    "Only carefully {gerund} src/app.py.",
    "By carefully {gerund} only src/app.py.",
    "During {gerund} only src/app.py.",
    "After {gerund} only src/app.py.",
    "In {gerund} only src/app.py.",
    "The {gerund} only src/app.py.",
    "Don’t edit config.py while {gerund} only src/app.py.",
    "Doesn’t edit config.py while {gerund} only src/app.py.",
    "Didn’t edit config.py while {gerund} only src/app.py.",
    "Isn’t editing config.py while {gerund} only src/app.py.",
    "Aren’t editing config.py while {gerund} only src/app.py.",
    "Wasn’t editing config.py while {gerund} only src/app.py.",
    "Weren’t editing config.py while {gerund} only src/app.py.",
    "Couldn’t edit config.py while {gerund} only src/app.py.",
    "Mightn’t edit config.py while {gerund} only src/app.py.",
    "Mustn’t edit config.py while {gerund} only src/app.py.",
    "Shouldn’t edit config.py while {gerund} only src/app.py.",
    "Wouldn’t edit config.py while {gerund} only src/app.py.",
    "Haven’t edited config.py while {gerund} only src/app.py.",
    "Hasn’t edited config.py while {gerund} only src/app.py.",
    "Hadn’t edited config.py while {gerund} only src/app.py.",
    "Shan’t edit config.py while {gerund} only src/app.py.",
    "Cannot edit config.py while {gerund} only src/app.py.",
    "Won’t edit config.py while {gerund} only src/app.py.",
    "Can’t edit config.py while {gerund} only src/app.py.",
    "I am {gerund} only src/app.py.",
    "Only to {gerund} src/app.py.",
    "Only being {gerund} src/app.py.",
    "Only have {gerund} src/app.py.",
    "Only will {gerund} src/app.py.",
    "Only can {gerund} src/app.py.",
    "Only should {gerund} src/app.py.",
    "Only might {gerund} src/app.py.",
    "Could only {gerund} src/app.py.",
    "Would only {gerund} src/app.py.",
    "Should only {gerund} src/app.py.",
    "Must only {gerund} src/app.py.",
    "Can only {gerund} src/app.py.",
    "Will only {gerund} src/app.py.",
    "May only {gerund} src/app.py.",
    "Do not only {gerund} src/app.py.",
    "Never only {gerund} src/app.py.",
    "Not only {gerund} src/app.py is allowed.",
    "If requested, {gerund} only src/app.py.",
    "When requested, {gerund} only src/app.py.",
    "Unless requested, {gerund} only src/app.py.",
    "{gerund} only src/app.py if needed.",
    "{gerund} only src/app.py when needed.",
    "{gerund} only src/app.py unless required.",
    "Only {past} src/app.py.",
    "By {past} only src/app.py.",
    "Is only {gerund} src/app.py allowed?",
    "Only {gerund} src/app.py?",
    'Example: "{gerund} only src/app.py."',
    "Example: '{gerund} only src/app.py.'",
    "Example: `{gerund} only src/app.py.`",
    "```text\n{gerund} only src/app.py.\n```",
)

NEGATIVE_CASES = [
    (pattern.format(gerund=gerund, past=_PAST_FORMS[gerund]), pattern, gerund)
    for pattern in _NEGATIVE_PATTERNS
    for gerund in GERUNDS
]

_BASE_RESULTS_WITHOUT_GERUNDS = (
    ("Fix the bug in src/app.py.", None),
    ("Only edit src/app.py.", ("src/app.py",)),
    ("Only change a.py, b.py and README.md.", ("a.py", "b.py", "README.md")),
    ("Fix a.py. Change no other file.", ("a.py",)),
    ("Only modify src/a.py if needed.", None),
    ("Only edit ../src/app.py.", TaskFileScopeError),
    ("Read README.md. Change no other file.", None),
    ("Only edit the file above.", TaskFileScopeError),
    ("Do not edit a.py. Change no other file.", None),
    ("Update a.py and only edit b.py.", ("b.py",)),
    ("Only write answer.json and update docs/a.md.", ("answer.json",)),
    (r"Only edit C:\repo\src\app.py.", TaskFileScopeError),
    ('Only edit src/app.py. Example: "Only edit src/b.py."', ("src/app.py",)),
    ("Only edit Dockerfile.", ("Dockerfile",)),
    ("Write answer.json. Change no other file.", ("answer.json",)),
    ("Only edit `./src/app.py`.", ("src/app.py",)),
    ("Only edit a.py for b.py.", ("a.py",)),
    ("", None),
    ("Fix src/app.py. Only edit tests/test_app.py if needed.", None),
    ("Only edit src/app.py using https://example.com/?file=b.py.", ("src/app.py",)),
)

_GERUND_WORD = re.compile(r"\b(?:" + "|".join(GERUNDS) + r")\b", re.IGNORECASE)


@pytest.mark.parametrize(("label", "task", "expected"), POSITIVE_CASES,
                         ids=[case[0] for case in POSITIVE_CASES])
def test_closed_gerund_templates_scope_only_the_named_paths(label, task, expected):
    assert parse_task_file_scope(task) == expected


def test_negative_table_is_at_least_three_times_larger_than_positive_table():
    assert len(NEGATIVE_CASES) >= 3 * len(POSITIVE_CASES)


@pytest.mark.parametrize("task", (
    "editing only ../src/app.py.",
    "Fix the bug by only writing /etc/config.py.",
))
def test_unsupported_gerund_targets_keep_the_existing_degrade_signal(task):
    with pytest.raises(TaskFileScopeError):
        parse_task_file_scope(task)


@pytest.mark.parametrize("task", (
    "editing only src/app.py if needed.",
    "Only writing src/app.py when needed.",
    "Fix the bug by only editing src/app.py unless necessary.",
))
def test_conditional_gerund_rules_keep_the_existing_strict_degrade_path(task):
    with pytest.raises(TaskFileScopeError, match="conditional"):
        parse_task_file_scope(task, strict=True)


@pytest.mark.parametrize(("task", "pattern", "gerund"), NEGATIVE_CASES,
                         ids=[f"{gerund}-{index}" for index in range(len(_NEGATIVE_PATTERNS))
                              for gerund in GERUNDS])
def test_near_misses_do_not_become_file_scope_rules(task, pattern, gerund):
    assert parse_task_file_scope(task) is None


@pytest.mark.parametrize(("task", "base_result"), _BASE_RESULTS_WITHOUT_GERUNDS)
def test_text_without_any_supported_gerund_keeps_recorded_base_result(task, base_result):
    assert _GERUND_WORD.search(task) is None
    if base_result is TaskFileScopeError:
        with pytest.raises(TaskFileScopeError):
            parse_task_file_scope(task)
    else:
        assert parse_task_file_scope(task) == base_result
