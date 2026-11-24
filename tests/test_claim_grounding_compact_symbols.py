"""Compact claim grounding keeps every lookup result and cache boundary stable."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import kernel_claim_grounding as grounding
import kernel_parse_cache as parse_cache
import pytest


@pytest.fixture(autouse=True)
def clear_claim_and_parse_caches():
    parse_cache.clear()
    with grounding._AST_INDEX_CACHE_LOCK:
        grounding._AST_INDEX_CACHE.clear()
    yield
    parse_cache.clear()
    with grounding._AST_INDEX_CACHE_LOCK:
        grounding._AST_INDEX_CACHE.clear()


def _write_sources(root: Path) -> dict[str, str]:
    sources = {
        "a.py": "def unique():\n    return 1\n\nclass Container:\n    def method(self):\n        return unique()\n\ndef shared():\n    return 2\n",
        "b.py": "def shared():\n    return 3\n",
        "c.py": "VALUE = 4\n",
    }
    for name, source in sources.items():
        (root / name).write_text(source, encoding="utf-8")
    return sources


def test_symbol_table_stores_one_path_as_string_and_duplicates_as_sorted_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _write_sources(tmp_path)
    index = grounding._AstIndex(tmp_path)

    assert index.symbols["unique"] == "a.py"
    assert type(index.symbols["unique"]) is str
    assert index.symbols["Container.method"] == "a.py"
    assert index.symbols["shared"] == ("a.py", "b.py")
    assert type(index.symbols["shared"]) is tuple

    payload = {
        "schema": grounding._AST_PERSIST_SCHEMA,
        "symbols": {
            "unique": ["a.py"],
            "shared": ["a.py", "b.py"],
        },
    }
    restored = grounding._ast_symbols_from_payload(payload, files=tuple(sources))
    assert restored == {"unique": "a.py", "shared": ("a.py", "b.py")}

    _candidates, paths, files, signature = index._inventory()
    persisted = {}
    monkeypatch.setattr(grounding._turn0, "sources_unchanged", lambda *_args: True)
    monkeypatch.setattr(
        grounding._turn0,
        "write_payload",
        lambda _context, value: persisted.update(value) or None,
    )
    grounding._save_persisted_ast_symbols(
        object(), index, signature=signature, journal=None,
    )
    expected_persisted = {
        key: [value] if isinstance(value, str) else list(value)
        for key, value in sorted(index.symbols.items())
    }
    assert persisted["symbols"] == expected_persisted
    assert persisted["symbols"]["unique"] == ["a.py"]
    assert persisted["symbols"]["shared"] == ["a.py", "b.py"]
    assert paths == index.paths and files == index.files


def test_public_symbol_and_path_results_match_cached_and_scoped_indexes(
    tmp_path: Path,
) -> None:
    _write_sources(tmp_path)
    first = grounding.build_ast_index(tmp_path)
    cached = grounding.build_ast_index(tmp_path)
    scoped = grounding._ScopedAstIndex(tmp_path, None)

    for cited in ("unique", "shared", "missing_symbol"):
        expected = first.resolve_symbol(cited).to_dict()
        assert cached.resolve_symbol(cited).to_dict() == expected
    for cited in ("unique", "shared"):
        assert scoped.resolve_symbol(cited).exists
    assert not scoped.resolve_symbol("missing_symbol").exists
    for cited in ("a.py", "b.py", "missing.py"):
        expected = first.resolve_path(cited, tmp_path).to_dict()
        assert cached.resolve_path(cited, tmp_path).to_dict() == expected
        assert scoped.resolve_path(cited, tmp_path).to_dict() == expected

    assert grounding.resolve_claim(tmp_path, "unique", "symbol").to_dict() == (
        first.resolve_symbol("unique").to_dict()
    )


def test_parse_and_derived_build_counts_do_not_increase_on_same_turn_index_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        f"module_{index:03}.py": (
            f"# compact-cache-probe-{index}\ndef symbol_{index}(value):\n"
            f"    return value + {index}\n"
        )
        for index in range(120)
    }
    estimate_source = next(iter(sources.values()))
    estimate = parse_cache._estimate_tree_retained_bytes(
        estimate_source.encode("utf-8"), parse_cache.ast.parse(estimate_source),
    )
    monkeypatch.setattr(parse_cache, "MAX_TREE_RETAINED_BYTES", 8 * estimate)
    for name, source in sources.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    parsed: Counter[str] = Counter()
    rebuilt: Counter[str] = Counter()
    original_parse = parse_cache.ast.parse
    original_symbols = grounding._AstIndex._symbol_names

    def counted_parse(source, *args, **kwargs):
        if isinstance(source, str) and "compact-cache-probe-" in source:
            parsed[source] += 1
        return original_parse(source, *args, **kwargs)

    def counted_symbols(self, tree, relative, module):
        rebuilt[relative] += 1
        return original_symbols(self, tree, relative, module)

    monkeypatch.setattr(parse_cache.ast, "parse", counted_parse)
    monkeypatch.setattr(grounding._AstIndex, "_symbol_names", counted_symbols)
    first = grounding._AstIndex(tmp_path)
    first_results = {
        symbol: first.resolve_symbol(symbol).to_dict()
        for symbol in ("symbol_000", "symbol_059", "symbol_119")
    }
    second = grounding._AstIndex(tmp_path)
    second_results = {
        symbol: second.resolve_symbol(symbol).to_dict()
        for symbol in ("symbol_000", "symbol_059", "symbol_119")
    }
    for name, source in sources.items():
        first._file_symbols(name, source)
    for name, source in sources.items():
        first._file_symbols(name, source)

    assert first_results == second_results
    assert len(parsed) == len(rebuilt) == len(sources)
    assert set(parsed.values()) == {1}
    assert set(rebuilt) == set(sources)
    assert set(rebuilt.values()) == {1}
    assert parse_cache.stats()["derived_misses"] == len(sources)
    assert parse_cache.stats()["derived_hits"] == 2 * len(sources)
    assert parse_cache.stats()["tree_retained_bytes"] <= 8 * estimate
