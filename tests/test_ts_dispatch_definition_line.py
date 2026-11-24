"""Definition-line lookup stays identical without compiling once per AST node."""
from __future__ import annotations

import ast
import random
import re

import pytest

from repo_intel_core import ts_dispatch


_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _old_definition_line(source_lines, node):
    fallback = int(getattr(node, "lineno", 1) or 1)
    name = str(getattr(node, "name", ""))
    keyword = "class" if isinstance(node, ast.ClassDef) else "def"
    decorators = getattr(node, "decorator_list", ())
    decorator_lines = [int(getattr(item, "lineno", fallback) or fallback) for item in decorators]
    first_line = min(decorator_lines, default=fallback)
    body = getattr(node, "body", ())
    last_line = int(getattr(body[0], "lineno", fallback) or fallback) if body else fallback
    declaration = re.compile(rf"^[ \t]*(?:async[ \t]+)?{keyword}[ \t]+{re.escape(name)}\b")
    for line_number in range(first_line, last_line + 1):
        if line_number <= len(source_lines) and declaration.match(source_lines[line_number - 1]):
            return line_number
    return fallback


def _random_source(rng: random.Random, index: int) -> str:
    names = ("work", "café", "λambda", "变量", "work_item", "naïve")
    name = f"{names[index % len(names)]}_{index}"
    keyword = "class" if index % 5 == 0 else "def"
    asynchronous = keyword == "def" and index % 2 == 0
    keyword_text = "async def" if asynchronous else keyword
    if asynchronous:
        keyword_text = "async" + rng.choice((" ", "  ", "\t", " \t ")) + "def"
    separator = rng.choice((" ", "  ", "\t", " \t "))
    after_name = rng.choice(("", " ", "  ", "\t"))
    parent = rng.choice(("", "if True:\n", "class Outer:\n    pass\n"))
    indent = "    " if parent else ""
    source = "# random definition-line parity source\n" + parent
    decorators = rng.randrange(4)
    decorator_text = ("@decorate", "@factory.method", "@decorate(value = 1)")
    for _ in range(decorators):
        source += indent + rng.choice(decorator_text) + "\n"
    source += indent + keyword_text + separator + name + after_name
    source += "(value):\n" if keyword == "def" else "(Base):\n"
    body_indent = indent + "    "
    if keyword == "class":
        source += body_indent + "def nested():\n"
        source += body_indent + "    return value\n"
    else:
        source += body_indent + rng.choice(("pass\n", "return value\n", "value = 1\n"))
    return source


def test_definition_line_and_outlines_match_previous_regex_for_200_random_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(0xC0FFEE)
    coverage = {"decorated": 0, "async": 0, "unicode": 0, "spacing": 0}
    for index in range(200):
        source = _random_source(rng, index)
        source_lines = source.splitlines()
        tree = ast.parse(source)
        nodes = [node for node in ast.walk(tree) if isinstance(node, _DEFINITION_NODES)]
        assert nodes, source
        with monkeypatch.context() as patch:
            patch.setattr(ts_dispatch, "_python_ast_definition_line", _old_definition_line)
            old_outline = ts_dispatch._python_ast_fallback(source, f"module_{index}.py", "extension")
        new_outline = ts_dispatch._python_ast_fallback(source, f"module_{index}.py", "extension")
        assert new_outline == old_outline, (index, source)
        for node in nodes:
            assert ts_dispatch._python_ast_definition_line(source_lines, node) == (
                _old_definition_line(source_lines, node)
            ), (index, source, node.name)
            coverage["decorated"] += bool(getattr(node, "decorator_list", ()))
            coverage["async"] += isinstance(node, ast.AsyncFunctionDef)
            coverage["unicode"] += any(ord(character) > 127 for character in node.name)
            declaration = source_lines[node.lineno - 1]
            coverage["spacing"] += "\t" in declaration or "  " in declaration

    assert all(coverage.values()), coverage


def test_definition_line_does_not_compile_a_regex_for_each_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "".join(f"def function_{index}(value):\n    return value\n" for index in range(40))
    source_lines = source.splitlines()
    nodes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)]
    expected = [_old_definition_line(source_lines, node) for node in nodes]

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("definition-line matching must not compile a per-node regex")

    monkeypatch.setattr(ts_dispatch.re, "compile", unexpected_compile)
    assert [ts_dispatch._python_ast_definition_line(source_lines, node) for node in nodes] == expected
