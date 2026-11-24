"""Shared boot parses stay read-only and bounded by retained source bytes."""
from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kernel_coordinated_change_ledger as ledger
import kernel_parse_cache as cache
import pytest
from symbol_registry import SymbolRegistry


@pytest.fixture(autouse=True)
def fresh_cache():
    cache.clear()
    yield
    cache.clear()


def _estimated_bytes(source: str) -> int:
    blocks_before = cache._allocated_blocks()
    tree = ast.parse(source)
    blocks_after = cache._allocated_blocks() if blocks_before is not None else None
    return cache._estimate_tree_retained_bytes(
        source.encode("utf-8", "surrogatepass"), tree,
        max(0, blocks_after - blocks_before)
        if blocks_before is not None and blocks_after is not None else 0,
    )


@pytest.mark.parametrize("registry_first", [False, True])
def test_package_parsed_once_across_ledger_and_registry(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch, registry_first: bool,
) -> None:
    # Both orderings use identical bytes. Give each its own cold persistent index.
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path_factory.mktemp("parse-cache-home")))
    package = tmp_path / "package"
    package.mkdir()
    sources = {
        f"def function_{index}(value, /, *, factor=2):\n"
        f"    return value * factor + {index}\n": index
        for index in range(600)
    }
    for source, index in sources.items():
        (package / f"module_{index:03}.py").write_text(source, encoding="utf-8")
    counts: Counter[str] = Counter()
    original_parse = ast.parse

    def counted_parse(source, *args, **kwargs):
        if isinstance(source, str) and source in sources:
            counts[source] += 1
        return original_parse(source, *args, **kwargs)

    monkeypatch.setattr(cache.ast, "parse", counted_parse)
    registry = SymbolRegistry()
    # Drive the real scan but bypass its repository/disk result caches, so a
    # persisted hit cannot conceal missing sharing between the AST consumers.
    if registry_first:
        registry.build(tmp_path)
        scanned = ledger._scan_repository(tmp_path)
    else:
        scanned = ledger._scan_repository(tmp_path)
        registry.build(tmp_path)

    assert registry.files_scanned == len(sources)
    assert len(scanned.definitions) == len(sources)
    assert len(counts) == len(sources)
    assert set(counts.values()) == {1}, (
        f"expected one parse per module; observed {Counter(counts.values())}"
    )
    assert registry.where("function_599")[0]["signature"] == (
        "def function_599(value, /, *, factor)"
    )


@pytest.mark.parametrize("literal", ["x", "é"])
def test_source_accounting_evicts_with_retained_byte_limit(
    monkeypatch: pytest.MonkeyPatch, literal: str,
) -> None:
    sources = [f"v = '{literal}{index}'\n" for index in range(3)]
    size = len(sources[0].encode("utf-8"))
    estimate = 4096
    monkeypatch.setattr(
        cache, "_estimate_tree_retained_bytes",
        lambda _source, _tree, _allocated_blocks=0: estimate,
    )
    monkeypatch.setattr(cache, "MAX_TREE_RETAINED_BYTES", 2 * estimate)
    first = cache.parse(sources[0], "first.py")
    second = cache.parse(sources[1], "second.py")
    assert cache.parse(sources[0], "first.py") is first  # refresh LRU order
    cache.parse(sources[2], "third.py")

    assert cache.stats()["trees"] == 2
    assert cache.stats()["tree_source_bytes"] == 2 * size
    assert cache.stats()["tree_retained_bytes"] == 2 * estimate
    assert cache.parse(sources[0], "first.py") is first
    assert cache.parse(sources[1], "second.py") is not second
    assert cache.stats()["tree_source_bytes"] == 2 * size


def test_retained_byte_counter_tracks_estimated_retained_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "def calculate(value):\n    return value + 1\n"
    estimate = 8192
    monkeypatch.setattr(
        cache, "_estimate_tree_retained_bytes",
        lambda _source, _tree, _allocated_blocks=0: estimate,
    )
    cache.parse(source, "calculate.py")
    source_bytes = len(source.encode("utf-8"))

    assert cache.stats()["tree_source_bytes"] == source_bytes
    assert cache.stats()["tree_retained_bytes"] == estimate
    assert cache.stats()["tree_retained_bytes"] >= cache.stats()["tree_source_bytes"]


def test_retained_byte_limit_evicts_lru_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [f"value = {index}\n" for index in range(3)]
    estimate = 1024
    monkeypatch.setattr(
        cache, "_estimate_tree_retained_bytes",
        lambda _source, _tree, _allocated_blocks=0: estimate,
    )
    budget = 2 * estimate
    monkeypatch.setattr(cache, "MAX_TREE_RETAINED_BYTES", budget, raising=False)
    first = cache.parse(sources[0], "first.py")
    second = cache.parse(sources[1], "second.py")
    assert cache.parse(sources[0], "first.py") is first
    cache.parse(sources[2], "third.py")

    assert cache.stats()["trees"] == 2
    assert cache.stats()["tree_retained_bytes"] == 2 * estimate
    assert cache.stats()["tree_retained_bytes"] <= budget
    assert cache.parse(sources[0], "first.py") is first
    assert cache.parse(sources[1], "second.py") is not second


def test_retained_oversize_tree_preserves_warm_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm_source = "def warm(): pass\n"
    oversized_source = warm_source + "".join(
        f"def additional_{index}():\n    pass\n" for index in range(100)
    )
    budget = 2 * _estimated_bytes(warm_source)
    monkeypatch.setattr(cache, "MAX_TREE_RETAINED_BYTES", budget, raising=False)
    warm = cache.parse(warm_source, "warm.py")

    first = cache.parse(oversized_source, "large.py")
    second = cache.parse(oversized_source, "large.py")

    assert first is not second
    assert ast.dump(first) == ast.dump(second)
    assert cache.parse(warm_source, "warm.py") is warm
    assert cache.stats()["trees"] == 1
    assert 0 < cache.stats()["tree_retained_bytes"] <= budget


def test_concurrent_retained_byte_accounting_stays_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        f"def function_{index:03}(value):\n    return value + {1000 + index}\n"
        for index in range(96)
    ]
    estimate = _estimated_bytes(sources[0])
    budget = 12 * estimate
    monkeypatch.setattr(cache, "MAX_TREE_RETAINED_BYTES", budget, raising=False)

    with ThreadPoolExecutor(max_workers=12) as pool:
        trees = list(pool.map(lambda row: cache.parse(row[1], f"{row[0]}.py"), enumerate(sources)))

    assert len(trees) == len(sources)
    assert all(isinstance(tree, ast.Module) for tree in trees)
    assert cache.stats()["tree_retained_bytes"] <= budget
    assert cache.stats()["tree_source_bytes"] <= cache.stats()["tree_retained_bytes"]


def test_oversized_parse_preserves_warm_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "v = 1\n"
    budget = 2 * _estimated_bytes(source)
    monkeypatch.setattr(cache, "MAX_TREE_RETAINED_BYTES", budget)
    warm = cache.parse(source, "warm.py")
    large = source + "".join(
        f"def additional_{index}(value):\n    return value + {index}\n"
        for index in range(100)
    )
    first = cache.parse(large, "large.py")
    second = cache.parse(large, "large.py")

    assert first is not second
    assert ast.dump(first) == ast.dump(second)
    assert cache.parse(source, "warm.py") is warm
    assert cache.stats()["trees"] == 1
    assert cache.stats()["tree_source_bytes"] == len(source)


def test_count_backstop_and_clear_release_byte_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache, "MAX_TREES", 2)
    for index in range(3):
        cache.parse("\n" * index, f"empty_{index}.py")
    assert cache.stats()["trees"] == 2
    assert cache.stats()["tree_source_bytes"] == 3
    cache.parse("x = 1\n", "nonempty.py")
    assert cache.stats()["tree_source_bytes"] == 8
    cache.clear()
    assert cache.stats()["trees"] == 0
    assert cache.stats()["tree_source_bytes"] == 0
    cache.parse("y = 2\n", "after_clear.py")
    assert cache.stats()["tree_source_bytes"] == 6


def test_ledger_validation_reuses_scan_trees_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "producer.py": "VALUE = 1\ndef produce():\n    return VALUE\n",
        "consumer.py": "from producer import produce\nresult = produce()\n",
    }
    before = {}
    for relative, source in sources.items():
        (tmp_path / relative).write_text(source, encoding="utf-8")
        tree = cache.parse(source, relative)
        before[relative] = ast.dump(tree, include_attributes=True)

    parsed = []
    original_parse = ast.parse

    def counted_parse(source, *args, **kwargs):
        parsed.append(source)
        return original_parse(source, *args, **kwargs)

    monkeypatch.setattr(cache.ast, "parse", counted_parse)
    scanned = ledger._scan_repository(tmp_path)
    registry = SymbolRegistry().build(tmp_path)
    for relative, source in sources.items():
        assert ledger._validate_file(source, relative) is None
    assert ledger._python_import_failures(
        tmp_path, {}, {}, changed_paths=sources,
    ) == ()
    assert parsed == [], "unchanged modules must reuse the warmed trees"
    assert registry.exists("produce")
    assert any(site.symbol == "produce" for site in scanned.definitions)
    for relative, source in sources.items():
        assert ast.dump(cache.parse(source, relative), include_attributes=True) == before[relative]


@pytest.mark.parametrize("source", ["def broken(:\n", "value = 1\x00\n", "value = '\ud800'\n"])
def test_parse_failures_keep_error_details_and_byte_accounting(source: str) -> None:
    with pytest.raises((SyntaxError, ValueError)) as expected:
        ast.parse(source, filename="invalid.py")
    for _ in range(2):
        with pytest.raises(type(expected.value)) as actual:
            cache.parse(source, "invalid.py")
        assert actual.value.args == expected.value.args
    assert cache.stats()["trees"] == 0
    assert cache.stats()["tree_source_bytes"] == 0


def test_valid_trees_share_content_across_filenames() -> None:
    first = cache.parse("value = 1\n", "one.py")
    changed = cache.parse("value = 2\n", "one.py")
    other = cache.parse("value = 1\n", "two.py")
    assert changed is not first
    assert other is first
    assert cache.parse("value = 1\n", "one.py") is first
