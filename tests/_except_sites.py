"""Symbol keys and inventory assertions for the T8 fail-loud tests."""
from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

ExceptSite = tuple[str, str, str, int]


def _is_broad(exception: ast.expr | None) -> bool:
    if isinstance(exception, ast.Tuple):
        return any(_is_broad(member) for member in exception.elts)
    return exception is None or ast.unparse(exception) in {
        "Exception", "BaseException", "builtins.Exception", "builtins.BaseException",
    }


def _broad_handlers(
    node: ast.AST, module: str, qualname: str = "", *, starred: bool = False,
) -> Iterator[tuple[str, str]]:
    match node:
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
            qualname = f"{qualname}.{name}" if qualname else name
        case ast.ExceptHandler(type=exception, name=alias) if _is_broad(exception):
            keyword = "except*" if starred else "except"
            caught = f" {ast.unparse(exception)}" if exception is not None else ""
            binding = f" as {alias}" if alias is not None else ""
            yield qualname or "<module>", f"{keyword}{caught}{binding}:"
        # The original census includes handlers in these two generated Python scripts.
        case ast.Assign(
            targets=[ast.Name(id="_PYTEST_RUNNER_SOURCE")], value=ast.Constant(value=str(source)),
        ) if module == "kernel_test_run_spans" and not qualname:
            yield from _broad_handlers(
                ast.parse(source, filename=f"{module}._PYTEST_RUNNER_SOURCE"), module, "_PYTEST_RUNNER_SOURCE",
            )
        case ast.Return(value=ast.BinOp(left=ast.Constant(value=str(source)), op=ast.Mod())) if (
            module, qualname
        ) == ("kernel_probe", "_wrapper_source"):
            # The probe substitutes reprs for these placeholders before executing the script.
            yield from _broad_handlers(
                ast.parse(source.replace("%r", "None"), filename=f"{module}.{qualname}"), module, qualname,
            )

    for child in ast.iter_child_nodes(node):
        yield from _broad_handlers(child, module, qualname, starred=isinstance(node, ast.TryStar))


def _broad_except_sites(paths: tuple[Path, ...]) -> set[ExceptSite]:
    """Return (module, qualname, except clause, one-based occurrence of that clause in the scope)."""
    sites: set[ExceptSite] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        occurrences: Counter[tuple[str, str]] = Counter()
        for qualname, clause in _broad_handlers(tree, path.stem):
            occurrences[qualname, clause] += 1
            sites.add((path.stem, qualname, clause, occurrences[qualname, clause]))
    return sites


def assert_triaged(found: set[ExceptSite], expected: set[ExceptSite]) -> None:
    assert found == expected, f"untriaged: {sorted(found - expected)}; stale: {sorted(expected - found)}"


def assert_site_tags(test_file: Path, found: set[ExceptSite]) -> None:
    source = test_file.read_text(encoding="utf-8")
    tags = set(re.findall(r"\bT8-(?:\[[^\]\n]*\]?|[A-Z0-9_]+-\d+)", source))
    assert tags, f"expected T8 site tags in {test_file.name}"
    known = {f"T8-[{module} | {name} | {clause} | {index}]" for module, name, clause, index in found}
    assert tags <= known, f"T8 tags do not name broad except sites: {sorted(tags - known)}"


def shifted_module_paths(paths: tuple[Path, ...], tmp_path: Path) -> tuple[Path, ...]:
    copies = tuple(tmp_path / path.name for path in paths)
    for path, copy in zip(paths, copies, strict=True):
        source = path.read_text(encoding="utf-8")
        copy.write_text("\n" * 5 + "# Unrelated edit above all exception handlers.\n" + source, encoding="utf-8")
    return copies


UNREVIEWED_HANDLER = "\n\ndef _t8_unreviewed():\n    try:\n        pass\n    except Exception:\n        pass\n"


def edited_module_paths(
    paths: tuple[Path, ...], tmp_path: Path, module: str, before: str | None, after: str,
) -> tuple[Path, ...]:
    original, = (path for path in paths if path.stem == module)
    source = original.read_text(encoding="utf-8")
    if before is None:
        changed = source + after
    else:
        assert source.count(before) == 1, f"expected one occurrence of {before!r} in {original.name}"
        changed = source.replace(before, after)
    assert changed != source
    copy = tmp_path / original.name
    copy.write_text(changed, encoding="utf-8")
    return tuple(copy if path == original else path for path in paths)
