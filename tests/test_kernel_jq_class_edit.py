import ast
import pathlib
import textwrap

import pytest

import kernel_jq_class_edit as edit
from kernel_jq_class_edit import prepare_class_edit


def test_prepare_class_edit_preserves_whitespace_only_literal_lines(tmp_path):
    current = "class Parent:\n    class Target:\n        value = 'old'\n"
    replacement = (
        '        class Target:\n'
        '            value = """first\n'
        '                \n'
        '            second\n'
        '            """\n'
    )

    plan = prepare_class_edit(
        current, "Target", replacement, path=pathlib.Path(tmp_path / "sample.py"),
    )

    assert plan.handled is True
    assert plan.rejection is None
    assert plan.candidate is not None
    assert plan.candidate.startswith(
        "class Parent:\n    class Target:\n        value = \"\"\"first\n"
    )
    value = ast.parse(plan.candidate).body[0].body[0].body[0].value.value
    assert value == "first\n                \n        second\n        "


def test_replace_fn_class_edit_uses_whitespace_preserving_normalization(
    tmp_path, monkeypatch,
):
    from stateframe.engine.probes import react_repair

    monkeypatch.setenv("STATEFRAME_JQ_CLASS_EDIT", "on")
    monkeypatch.setenv("STATEFRAME_TOKENLAB_EDIT_HARDENING", "off")
    monkeypatch.setenv("STATEFRAME_EDIT_NORMALIZATION", "off")
    monkeypatch.setenv("STATEFRAME_EFFECT_BOUNDARY", "off")
    path = tmp_path / "sample.py"
    path.write_text(
        "class Parent:\n    class Target:\n        value = 'old'\n",
        encoding="utf-8",
    )
    replacement = (
        '        class Target:\n'
        '            value = """first\n'
        '                \n'
        '            second\n'
        '            """\n'
    )

    result = react_repair.replace_fn(path, "Target", replacement, root=tmp_path)

    assert result.startswith("edit applied.")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    value = tree.body[0].body[0].body[0].value.value
    assert value.splitlines()[1] == "                "


_BLANK = " " * 16


@pytest.mark.parametrize("prefix", ["", "r", "b", "f", "rf"])
@pytest.mark.parametrize("quote", ['"""', "'''"])
def test_every_multiline_string_kind_keeps_its_blank_line(tmp_path, prefix, quote):
    value = f"{prefix}{quote}first\n{_BLANK}\n            second{quote}"
    replacement = f"        class Target:\n            value = {value}\n"

    plan = prepare_class_edit("class Target:\n    value = 1\n", "Target", replacement,
                              path=tmp_path / "sample.py")

    assert plan.rejection is None
    assert f"first\n{_BLANK}\n" in plan.candidate


def test_nested_fstring_keeps_its_blank_line(tmp_path):
    replacement = ('        class Target:\n            value = f"""a {f"{1}"}\n' + _BLANK
                   + '\n            b"""\n')

    plan = prepare_class_edit("class Target:\n    value = 1\n", "Target", replacement,
                              path=tmp_path / "sample.py")

    assert f"\n{_BLANK}\n" in plan.candidate


@pytest.mark.parametrize("odd", ["\x0c", "\x1c", "\u2028", "\x85"])
def test_other_line_terminators_do_not_shift_the_kept_lines(odd):
    source = f'    x = "a{odd}b"\n    y = """p\n{_BLANK}\n    q"""\n    \n    z = 1\n'

    result = edit._dedent_source_preserving_literal_blank_lines(source)

    assert result == f'x = "a{odd}b"\ny = """p\n{_BLANK}\nq"""\n\nz = 1\n'


def test_blank_code_lines_are_normalized_as_before():
    source = "    class A:\n        \n        x = 1\n            \n        y = 'a'\n"
    assert edit._dedent_source_preserving_literal_blank_lines(source) == textwrap.dedent(source)


def test_unfinished_source_keeps_legacy_dedent():
    for source in ("    x = '''open\n        \n", "    x = (\n        \n", "    f\"{\n        \n"):
        assert edit._dedent_source_preserving_literal_blank_lines(source) == textwrap.dedent(source)
