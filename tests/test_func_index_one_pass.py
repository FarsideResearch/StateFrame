from __future__ import annotations

import ast
import hashlib
import json
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
