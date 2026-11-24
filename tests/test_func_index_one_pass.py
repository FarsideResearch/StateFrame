from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))

TASK_WORDS = (
    "amber", "birch", "cobalt", "delta", "elm", "flint", "grove", "harbor",
    "indigo", "juniper", "kestrel", "lilac", "maple", "nectar", "orchid",
    "pebble", "quartz", "ripple", "silver", "thistle",
)
TIP_DIGESTS = {
    40: (
        "4876915edb6468965f33ec2fb703218d29987958549ca71bfb5c221f32ea62b5",
        "31c1aae8aae7e104a8a8a32dea5b2d4024b3f544b30095c19b7b270985a73ebf",
    ),
    875: (
        "5ff380bb8bcd34c8d2dd8e6ddf714845acecbcc21a0004880b6a94a9e1ec7f23",
        "eb20e589401f6c8599ab652873a9dc7c780b44c85146e6006c4227d89fd41de1",
    ),
}


def _sources(count: int) -> dict[str, str]:
    sources = {"target.py": "def target(value):\n    return value\n"}
    for number in range(count):
        word = TASK_WORDS[number % len(TASK_WORDS)]
        sources[f"app_{number:04d}.py"] = (
            "from target import target\n"
            f"VALUE_{number:04d} = {number}\n"
            f"def use_{word}(value, offset=0):\n"
            "    return target(value) + offset\n"
        )
    return sources


def _normalize(value):
    import dataclasses

    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _query_wire(func_index, index) -> bytes:
    result = []
    for number in range(20):
        query = f"locate use {TASK_WORDS[number]} arithmetic value update"
        pool, ranked = func_index.function_pool(query, "arith_flip", index)
        menu, membership = func_index.file_menu_from_functions(ranked, k=8)
        result.append(json.dumps({
            "pool": [item.func_id for item in pool],
            "menu": menu,
            "membership": membership,
        }, sort_keys=True, separators=(",", ":")).encode())
    return b"\0".join(result)


@pytest.mark.parametrize("count", [40, 875])
def test_function_index_parses_once_and_keeps_tip_results(monkeypatch, count):
    import func_index

    sources = _sources(count)
    parsed = 0
    original = ast.parse

    def count_parse(*args, **kwargs):
        nonlocal parsed
        parsed += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(func_index.ast, "parse", count_parse)
    index = func_index.build_func_index(sources)
    expected_index, expected_queries = TIP_DIGESTS[count]
    index_payload = _normalize({
        "records": index.records, "df": index.df, "file_idx": index.file_idx,
        "file_df": index.file_df, "file_n": index.file_n, "adj": index.adj,
    })
    actual_index = hashlib.sha256(json.dumps(
        index_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    actual_queries = hashlib.sha256(_query_wire(func_index, index)).hexdigest()
    assert actual_index == expected_index
    assert actual_queries == expected_queries
    assert parsed == len(sources)

    parsed = 0
    rebuilt = func_index.build_func_index(sources)
    assert parsed == len(sources)
    assert _normalize(rebuilt) == _normalize(index)


def test_function_records_use_slotted_storage():
    import func_index

    record = func_index.FuncRecord(
        func_id="module.py::compute::1", path="module.py", name="compute",
        qualname="compute", lineno=1, end_lineno=1, loc=1, is_module=False,
    )
    assert not hasattr(record, "__dict__")


def test_import_adjacency_keeps_duplicate_names_and_namespace_imports():
    import func_index

    files = {
        "namespace/left.py": "from .right import *\ndef left(value):\n    return value\n",
        "namespace/right.py": "from .left import *\ndef right(value):\n    return value\n",
        "pkg_one/collision.py": "def one(value):\n    return value\n",
        "pkg_two/collision.py": "def two(value):\n    return value\n",
        "consumer.py": "from collision import *\n",
        "broken.py": "def broken(:\n    return 1\n",
        "bom.py": "\ufeffdef bom(value):\n    return value\n",
        "odd_encoding.py": (
            "# coding: latin-1\nLABEL = 'café ñ'\n"
            "def odd(value):\n    return value + len(LABEL)\n"
        ),
    }
    index = func_index.build_func_index(files)
    assert index.adj["namespace/left.py"] == {"namespace/right.py"}
    assert index.adj["namespace/right.py"] == {"namespace/left.py"}
    assert index.adj["consumer.py"] == {"pkg_one/collision.py", "pkg_two/collision.py"}
    assert index.adj["pkg_one/collision.py"] == {"consumer.py"}
    assert index.adj["pkg_two/collision.py"] == {"consumer.py"}
    for path in ("broken.py", "bom.py"):
        assert [row.qualname for row in index.by_path[path]] == ["<module>"]
    assert [row.qualname for row in index.by_path["odd_encoding.py"]] == ["odd", "<module>"]


def test_rebuild_uses_renamed_deleted_and_edited_sources():
    import func_index

    first = func_index.build_func_index({
        "old_name.py": "def old_name(value):\n    return value\n",
        "deleted.py": "def deleted(value):\n    return value\n",
        "edited.py": "def alpha(value):\n    return value\n",
    })
    second = func_index.build_func_index({
        "renamed.py": "def old_name(value):\n    return value\n",
        "edited.py": "def bravo(value):\n    return value\n",
    })
    assert "old_name.py" in first.by_path and "old_name.py" not in second.by_path
    assert "deleted.py" in first.by_path and "deleted.py" not in second.by_path
    assert [row.name for row in second.by_path["edited.py"]] == ["bravo", "<module>"]


# The deliberately slow reference below walks the syntax tree once for file facts
# and once for every function's bag. It keeps the indexing contract independent of
# the production traversal strategy.
_REFERENCE_PATTERNS = {
    "cmp_flip": r" >= | <= | > | < ",
    "bool_flip": r" and | or ",
    "none_flip": r" is None| is not None",
    "arith_flip": r"\w [+\-] \w",
    "not_flip": r"\bnot\b",
}


def _reference_words(value):
    out = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value):
        for part in re.split(r"_+", token):
            for word in re.findall(r"[A-Z]?[a-z]+|[A-Z]{2,}", part):
                if len(word) >= 3:
                    out.add(word.lower())
    return out


def _reference_add(bag, values, weight):
    for value in values:
        bag[value] = max(bag.get(value, 0.0), weight)


def _reference_constructs(source):
    return {name: len(re.findall(pattern, source))
            for name, pattern in _REFERENCE_PATTERNS.items()}


def _reference_functions(tree):
    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = prefix + child.name
                yield child, qualname
                yield from visit(child, qualname + ".")
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, prefix + child.name + ".")
            else:
                yield from visit(child, prefix)

    yield from visit(tree, "")


def _reference_function_bag(function):
    bag = {}
    _reference_add(bag, _reference_words(function.name), 3.0)
    args = function.args
    for arg in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs):
        _reference_add(bag, _reference_words(arg.arg), 1.5)

    # Revisit the entire function subtree for each record, as the original slow
    # per-function formulation did. Nested definitions intentionally contribute
    # to enclosing functions as well as receiving their own records.
    for node in ast.walk(function):
        if node is function:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _reference_add(bag, _reference_words(node.name), 1.5)
        elif isinstance(node, ast.ClassDef):
            _reference_add(bag, _reference_words(node.name), 1.5)
        elif isinstance(node, ast.Name):
            _reference_add(bag, _reference_words(node.id), 1.0)
        elif isinstance(node, ast.Attribute):
            _reference_add(bag, _reference_words(node.attr), 1.2)
    return bag


def _reference_record(**values):
    return values


def _reference_file(path, source):
    path_words = _reference_words(path)
    lines = source.splitlines()
    file_bag = {}
    _reference_add(file_bag, path_words, 3.0)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return file_bag, [], [
            _reference_record(
                func_id=f"{path}::<module>::1", path=path, name="<module>",
                qualname="<module>", lineno=1, end_lineno=max(1, len(lines)),
                loc=max(1, len(lines)), is_module=True,
                ident_bag=dict(file_bag), doc_bag={},
                construct_counts=_reference_constructs(source),
                parent_path_words=path_words, parent_import_words=set(),
            )
        ], True

    imports = set()
    modules = set()
    module_terms = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _reference_add(file_bag, _reference_words(node.name), 2.5)
            _reference_add(file_bag,
                           _reference_words((ast.get_docstring(node) or "")[:300]),
                           1.5)
        elif isinstance(node, ast.ClassDef):
            class_words = _reference_words(node.name)
            _reference_add(file_bag, class_words, 2.5)
            _reference_add(file_bag,
                           _reference_words((ast.get_docstring(node) or "")[:300]),
                           1.5)
            module_terms.append((class_words, 2.5))
        elif isinstance(node, ast.Name):
            name_words = _reference_words(node.id)
            _reference_add(file_bag, name_words, 1.0)
            module_terms.append((name_words, 0.8))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.update(_reference_words(alias.name))
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.update(_reference_words(node.module))
            modules.add(node.module.split(".")[0])

    _reference_add(file_bag,
                   _reference_words((ast.get_docstring(tree) or "")[:300]), 2.0)

    records = []
    covered = set()
    for function, qualname in _reference_functions(tree):
        start = function.lineno
        end = getattr(function, "end_lineno", function.lineno)
        doc_bag = {}
        _reference_add(doc_bag,
                       _reference_words((ast.get_docstring(function) or "")[:400]),
                       1.5)
        records.append(_reference_record(
            func_id=f"{path}::{qualname}::{start}", path=path,
            name=function.name, qualname=qualname, lineno=start,
            end_lineno=end, loc=max(1, end - start + 1), is_module=False,
            ident_bag=_reference_function_bag(function), doc_bag=doc_bag,
            construct_counts=_reference_constructs(
                "\n".join(lines[start - 1:end])),
            parent_path_words=path_words, parent_import_words=imports,
        ))
        if function.col_offset == 0:
            covered.update(range(start, end + 1))

    module_lines = [line for number, line in enumerate(lines, 1)
                    if number not in covered]
    module_bag = {}
    _reference_add(module_bag, imports, 1.0)
    for tokens, weight in module_terms:
        _reference_add(module_bag, tokens, weight)
    module_doc = {}
    _reference_add(module_doc,
                   _reference_words((ast.get_docstring(tree) or "")[:400]), 2.0)
    records.append(_reference_record(
        func_id=f"{path}::<module>::1", path=path, name="<module>",
        qualname="<module>", lineno=1, end_lineno=max(1, len(lines)),
        loc=max(1, len(module_lines)), is_module=True,
        ident_bag=module_bag, doc_bag=module_doc,
        construct_counts=_reference_constructs("\n".join(module_lines)),
        parent_path_words=path_words, parent_import_words=imports,
    ))
    return file_bag, sorted(modules), records, False


def _reference_index(files):
    file_idx = {}
    file_df = {}
    paths_by_module = {}
    for path in files:
        module = Path(path).stem
        existing = paths_by_module.get(module)
        if existing is None:
            paths_by_module[module] = path
        elif isinstance(existing, list):
            existing.append(path)
        else:
            paths_by_module[module] = [existing, path]

    adjacency = {path: set() for path in files}
    records = []
    for path, source in files.items():
        bag, imports, file_records, syntax_error = _reference_file(path, source)
        file_idx[path] = bag
        if not syntax_error:
            for word in bag:
                file_df[word] = file_df.get(word, 0) + 1
        records.extend(file_records)
        if syntax_error:
            continue
        for module in imports:
            neighbors = paths_by_module.get(module, ())
            if isinstance(neighbors, str):
                neighbors = (neighbors,)
            for neighbor in neighbors:
                if neighbor != path:
                    adjacency[path].add(neighbor)
                    adjacency[neighbor].add(path)

    def record_bag(record):
        result = dict(record["ident_bag"])
        for word, weight in record["doc_bag"].items():
            result[word] = max(result.get(word, 0.0), weight)
        return result

    df = {}
    by_path = {}
    for record in records:
        for word in record_bag(record):
            df[word] = df.get(word, 0) + 1
        by_path.setdefault(record["path"], []).append(record)
    return {
        "records": records, "df": df, "n": len(records),
        "file_idx": file_idx, "file_df": file_df, "file_n": len(files),
        "adj": adjacency, "by_path": by_path,
    }


def _actual_index(index):
    return {
        "records": index.records, "df": index.df, "n": index.n,
        "file_idx": index.file_idx, "file_df": index.file_df,
        "file_n": index.file_n, "adj": index.adj, "by_path": index.by_path,
    }


def _actual_and_slow_reference(files):
    import func_index

    actual = _normalize(_actual_index(func_index.build_func_index(files)))
    return actual, _normalize(_reference_index(files))


class TestSlowReferenceComparison:
    def test_nested_classes_decorators_async_and_expressions_match(self):
        source = '''
"""A module docstring with acorn details."""
from collections import deque

@decorate_with(LEVEL)
async def outer_service(payload, /, retries=2, *, timeout=5):
    """An outer service for queued payload work."""
    transform = lambda item: item.value.strip()
    selected = [transform(item) for item in payload if item.ready]

    class LocalWorker:
        """A local worker class."""
        @classmethod
        def create(cls, values):
            return deque(values)

    async def inner_worker(value):
        """An inner worker for the service."""
        return await value.run()

    if retries:
        def conditional_worker(value):
            return inner_worker(value)

    return selected
'''
        actual, expected = _actual_and_slow_reference({"complex_module.py": source})
        assert actual == expected


class TestFallbackReferenceComparison:
    def test_syntax_error_non_ascii_and_empty_sources_match(self):
        files = {
            "broken.py": "def unfinished(:\n    return 3\n",
            "unicode_names.py": (
                '"""Crème brûlée and 東京."""\n'
                'naïve_value = "façade 東京"\n'
                "def café_handler(entrée):\n"
                "    return entrée + len(naïve_value)\n"
            ),
            "empty_module.py": "",
        }
        actual, expected = _actual_and_slow_reference(files)
        assert actual == expected


class TestReverseFileOrderReferenceComparison:
    def test_reversed_file_order_matches_slow_reference(self):
        files = {
            "zeta.py": "def zulu(value):\n    return value\n",
            "alpha.py": "def alpha(value):\n    return value\n",
            "middle.py": "if READY:\n    def conditional():\n        return 1\n",
        }
        reversed_files = dict(reversed(list(files.items())))
        actual, expected = _actual_and_slow_reference(reversed_files)
        assert actual == expected


class TestLinearAstVisits:
    @staticmethod
    def _nested_source(depth):
        lines = ["def level_0(value):"]
        indentation = "    "
        for number in range(1, depth):
            lines.append(indentation + f"def level_{number}(value):")
            indentation += "    "
        lines.append(indentation + "return value")
        return "\n".join(lines) + "\n"

    def test_ast_child_visits_scale_linearly_with_nested_depth(self, monkeypatch):
        import func_index

        original = ast.iter_child_nodes

        def visits_for(depth):
            count = 0

            def counting_children(node):
                nonlocal count
                children = list(original(node))
                count += len(children)
                return iter(children)

            with monkeypatch.context() as patch:
                patch.setattr(ast, "iter_child_nodes", counting_children)
                func_index.build_func_index({
                    f"deep_{depth}.py": self._nested_source(depth),
                })
            return count

        shallow = visits_for(24)
        deep = visits_for(48)
        assert deep > shallow
        assert deep <= shallow * 2.7
