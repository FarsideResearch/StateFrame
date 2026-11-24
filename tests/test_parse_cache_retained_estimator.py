from __future__ import annotations

import ast
import gc
import random
import sys
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from time import process_time
from types import SimpleNamespace

import kernel_parse_cache as cache
import pytest


@pytest.fixture(autouse=True)
def fresh_cache():
    cache.clear()
    yield
    cache.clear()


def _measure_parse(source: str) -> tuple[ast.Module, int, int]:
    # The process-wide table of interned names may grow during one parse and is
    # then charged to it. It cannot grow in both, so keep the smaller measurement.
    first = _measure_parse_once(source)
    second = _measure_parse_once(source)
    return min(first, second, key=lambda row: row[1])


def _measure_parse_once(source: str) -> tuple[ast.Module, int, int]:
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    gc.collect()
    before_bytes = tracemalloc.get_traced_memory()[0]
    before_blocks = cache._allocated_blocks()
    tree = ast.parse(source)
    after_blocks = cache._allocated_blocks() if before_blocks is not None else None
    retained = tracemalloc.get_traced_memory()[0] - before_bytes
    if not tracing_before:
        tracemalloc.stop()
    blocks = (
        max(0, after_blocks - before_blocks)
        if before_blocks is not None and after_blocks is not None else 0
    )
    return tree, retained, blocks


def _random_module(rng: random.Random, index: int) -> str:
    lines = [f"BASE_{index} = {rng.randrange(10**9)}"]
    for block in range(rng.randint(2, 8)):
        name = f"item_{index}_{block}"
        kind = rng.randrange(5)
        if kind == 0:
            lines.append(f"{name} = [{rng.randrange(1000)} for value in range({rng.randrange(2, 20)})]")
        elif kind == 1:
            lines.extend((
                f"def {name}(value: int | None, factor: int = {rng.randrange(1, 8)}) -> str:",
                f"    return f'{{value!r}}-{{factor}}-{index}'",
            ))
        elif kind == 2:
            lines.append(f"{name} = lambda value: (value, {rng.randrange(1000)})")
        elif kind == 3:
            lines.append(f"{name} = {{'key': '{rng.randrange(100000)}', 'index': {index}}}")
        else:
            lines.append(f"{name} = f'value-{index}-{block}'")
    return "\n".join(lines) + "\n"


def _estimator_sources() -> dict[str, str]:
    dense = ";".join(f"value_{index} = {index}" for index in range(4_000))
    numeric_table = "ROWS = [\n" + "".join(
        f"    ({index}, {index + 1}, {index + 2}),\n" for index in range(2_000)
    ) + "]\n"
    decorators = "".join(
        f"@decorator_a\n@decorator_b\n@decorator_c\ndef function_{index}(value):\n    return value\n"
        for index in range(80)
    )
    match_source = "def choose(value):\n    match value:\n" + "".join(
        f"        case {index}:\n            return {index}\n" for index in range(80)
    ) + "        case _:\n            return None\n"
    typed_functions = "".join(
        f"def typed_{index}(value: dict[str, list[int]], *, key: str = 'x') -> tuple[int, ...]:\n"
        "    return tuple(value.get(key, ()))\n"
        for index in range(80)
    )
    small_functions = "".join(
        f"def f{index}(value):\n    return value + {index}\n"
        for index in range(100)
    )
    large_table = "TABLE = [" + ",".join(
        f"({index}, 'k{index}')" for index in range(300_000)
    ) + "]\n"
    huge_method = (
        "class Example:\n"
        + "".join(f"    def small_{index}(self): return {index}\n" for index in range(40))
        + "    def large(self):\n"
        + "".join(f"        value_{index} = call(a, b, {index})\n" for index in range(60_000))
        + "".join(f"    def trailing_{index}(self): return {index}\n" for index in range(40))
    )
    return {
        "dense-statements": dense,
        "literals": "TEXT = " + repr("payload-" * 25_000) + "\nBYTES = b'abc'\nNUMBER = 123456789012345678901234567890\n",
        "f-strings": "".join(
            f"MESSAGE_{index} = f'{{value_{index}!r:>24}} / {{index}}'\n" for index in range(100)
        ),
        "deep-nesting": "value = " + "not " * 300 + "True\n",
        "comments": "# comment-only source without AST payload\n" * 5_000,
        "docstrings": '"""' + ("Long documentation paragraph with plain words.\n" * 5_000) + '"""\n',
        "decorators": decorators,
        "lambdas": "factory = " + "lambda value: " * 120 + "value\n",
        "match": match_source,
        "comprehensions": (
            "values = [value * 2 for group in groups for value in group if value % 3]\n"
            "lookup = {key: value for key, value in pairs if key}\n"
        ),
        "type-hints": typed_functions,
        "numeric-table": numeric_table,
        "one-big-table-among-100-functions": small_functions + large_table + small_functions,
        "class-with-one-huge-method": huge_method,
        **{
            f"random-{index:03}": _random_module(random.Random(0x51A7E + index), index)
            for index in range(300)
        },
    }


def test_estimate_covers_ast_retained_memory_for_required_corpus() -> None:
    if cache._allocated_blocks() is None:
        pytest.skip("sys.getallocatedblocks is needed to measure this estimator")
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    measurements: dict[str, tuple[int, int]] = {}
    try:
        for label, source in _estimator_sources().items():
            data = source.encode("utf-8", "surrogatepass")
            gc.collect()
            before = tracemalloc.get_traced_memory()[0]
            blocks_before = cache._allocated_blocks()
            tree = ast.parse(source, filename=f"{label}.py")
            blocks_after = cache._allocated_blocks()
            retained = tracemalloc.get_traced_memory()[0] - before
            assert blocks_before is not None and blocks_after is not None
            estimate = cache._estimate_tree_retained_bytes(
                data, tree, max(0, blocks_after - blocks_before),
            )
            ratio = retained / max(estimate, 1)
            assert ratio <= 1.5, (
                f"{label}: actual={retained}, estimate={estimate}, "
                f"actual/estimate={ratio:.3f}"
            )
            if label != "comments":
                overestimate = estimate / max(retained, 1)
                assert overestimate <= 3, (
                    f"{label}: actual={retained}, estimate={estimate}, "
                    f"estimate/actual={overestimate:.3f}"
                )
            measurements[label] = retained, estimate
            del tree
            gc.collect()
    finally:
        if not tracing_before:
            tracemalloc.stop()

    assert len(measurements) == 314
    assert all(estimate >= cache.MIN_TREE_RETAINED_BYTES for _, estimate in measurements.values())
    worst_label = max(
        measurements,
        key=lambda label: measurements[label][0] / max(measurements[label][1], 1),
    )
    worst_actual, worst_estimate = measurements[worst_label]
    print(
        f"retained-estimate sources={len(measurements)} worst={worst_label} "
        f"actual={worst_actual} estimate={worst_estimate} "
        f"actual/estimate={worst_actual / max(worst_estimate, 1):.4f} "
        f"estimate/actual={worst_estimate / max(worst_actual, 1):.4f}"
    )


def test_estimate_cost_is_at_most_five_percent_of_parse_median() -> None:
    source = "".join(
        f"def function_{index:04}(value):\n    result = value + {index}\n    return result\n"
        for index in range(1_000)
    )
    data = source.encode("utf-8", "surrogatepass")
    tree = ast.parse(source)
    parse_samples = []
    estimate_samples = []
    parsed_trees = [tree]
    for _ in range(11):
        with cache.gc_paused():
            blocks_before = cache._allocated_blocks()
            started = process_time()
            parsed = ast.parse(source)
            parse_samples.append(process_time() - started)
            blocks_after = cache._allocated_blocks()
            measured_delta = (max(0, blocks_after - blocks_before)
                              if blocks_before is not None and blocks_after is not None else 0)
            started = process_time()
            cache._estimate_tree_retained_bytes(data, parsed, measured_delta)
            estimate_samples.append(process_time() - started)
            parsed_trees.append(parsed)

    parse_median = median(parse_samples)
    estimate_median = median(estimate_samples)
    ratio = estimate_median / max(parse_median, 1e-9)
    print(
        f"estimate-cost median={estimate_median:.6f}s parse-median={parse_median:.6f}s "
        f"ratio={ratio:.3%}"
    )
    assert ratio <= 0.05, (
        f"estimate median={estimate_median:.6f}s, parse median={parse_median:.6f}s, "
        f"ratio={ratio:.3%}"
    )


def test_six_estimator_mutants_are_rejected_by_the_adversarial_corpus() -> None:
    def within_limits(actual: int, estimate: int) -> bool:
        return (
            actual / max(estimate, 1) <= 1.5
            and estimate / max(actual, 1) <= 3
        )

    small_functions = "".join(
        f"def f{index}(value):\n    return value + {index}\n"
        for index in range(100)
    )
    table_source = small_functions + "TABLE = [" + ",".join(
        f"({index}, 'k{index}')" for index in range(300_000)
    ) + "]\n" + small_functions
    table, table_actual, _table_blocks = _measure_parse(table_source)
    table_data = table_source.encode("utf-8", "surrogatepass")
    leading = table.body[:8]
    sampled_estimate = sum(
        sys.getsizeof(node) + sys.getsizeof(vars(node)) for node in leading
    ) * len(table.body) // max(len(leading), 1)

    docstring = '"""' + ("Long documentation paragraph. " * 100_000) + '"""\n'
    doc_tree, doc_actual, doc_blocks = _measure_parse(docstring)
    doc_data = docstring.encode("utf-8", "surrogatepass")

    dense = ";".join(f"value_{index} = {index}" for index in range(4_000))
    _dense_tree, dense_actual, dense_blocks = _measure_parse(dense)
    dense_data = dense.encode("utf-8", "surrogatepass")

    rejected = {
        "source-only": not within_limits(table_actual, len(table_data)),
        "sample-leading-statements": not within_limits(table_actual, sampled_estimate),
        "block-only": not within_limits(doc_actual, doc_blocks * 96),
        "low-block-multiplier": not within_limits(
            dense_actual, dense_blocks * 32 + len(dense_data),
        ),
        "no-source-floor": not within_limits(table_actual, 512),
        "omit-literal-values": not within_limits(
            doc_actual, sum(sys.getsizeof(node) + sys.getsizeof(vars(node))
                            for node in ast.walk(doc_tree)),
        ),
    }
    assert rejected == {name: True for name in rejected}, rejected
    print(f"estimator mutants rejected={sum(rejected.values())}/{len(rejected)}")


def test_estimator_handles_deep_trees_without_recursion() -> None:
    expression: ast.expr = ast.Constant(value=1)
    for _ in range(10_000):
        expression = ast.UnaryOp(op=ast.Not(), operand=expression)
    tree = ast.Module(body=[ast.Expr(value=expression)], type_ignores=[])

    estimate = cache._estimate_tree_retained_bytes(b"", tree)

    assert estimate >= cache.MIN_TREE_RETAINED_BYTES


def test_source_floor_covers_large_constant_when_allocation_delta_is_lost() -> None:
    source = '"""' + ("documentation " * 200_000) + '"""\n'
    data = source.encode("utf-8", "surrogatepass")
    tree, actual, _ = _measure_parse(source)

    estimate = cache._estimate_tree_retained_bytes(data, tree, allocated_blocks=0)

    assert actual / max(estimate, 1) <= 1.5
    assert estimate / max(actual, 1) <= 3


def test_collection_during_parse_cannot_remove_source_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = '"""' + ("collected during parse " * 100_000) + '"""\n'
    cycles: list[list[object]] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(20_000):
            cycle: list[object] = []
            cycle.append(cycle)
            cycles.append(cycle)
        cycles.clear()

        collected: list[int] = []
        original_parse = ast.parse

        def collect_inside_parse(value: str, *args, **kwargs) -> ast.Module:
            tree = original_parse(value, *args, **kwargs)
            collected.append(gc.collect())
            return tree

        monkeypatch.setattr(cache.ast, "parse", collect_inside_parse)
        tracing_before = tracemalloc.is_tracing()
        if not tracing_before:
            tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        tree = cache.parse(source, "gc-during-parse.py")
        actual = tracemalloc.get_traced_memory()[0] - before
        estimate = cache.stats()["tree_retained_bytes"]
        if not tracing_before:
            tracemalloc.stop()

        assert collected and collected[0] > 0
        assert isinstance(tree, ast.Module)
        assert actual / max(estimate, 1) <= 1.5
        assert estimate / max(actual, 1) <= 3
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


def test_other_thread_freeing_during_parse_cannot_remove_source_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = '"""' + ("released from another thread " * 80_000) + '"""\n'
    held: list[list[object]] = [[object()] for _ in range(50_000)]
    real_parse = ast.parse
    parse_block_deltas: list[int] = []

    def parse_while_other_thread_frees(value: str, *args, **kwargs) -> ast.Module:
        before = cache._allocated_blocks()
        tree = real_parse(value, *args, **kwargs)
        releaser = threading.Thread(target=held.clear)
        releaser.start()
        releaser.join(timeout=10)
        assert not releaser.is_alive()
        after = cache._allocated_blocks()
        assert before is not None and after is not None
        parse_block_deltas.append(after - before)
        return tree

    monkeypatch.setattr(cache.ast, "parse", parse_while_other_thread_frees)
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tree = cache.parse(source, "thread-free-during-parse.py")
    actual = tracemalloc.get_traced_memory()[0] - before
    estimate = cache.stats()["tree_retained_bytes"]
    if not tracing_before:
        tracemalloc.stop()

    assert parse_block_deltas and parse_block_deltas[0] < 0
    assert isinstance(tree, ast.Module)
    assert actual / max(estimate, 1) <= 1.5
    assert estimate / max(actual, 1) <= 3


@pytest.mark.parametrize("counter_state", ["missing", "zero"])
def test_missing_or_zero_allocation_counter_uses_exact_fallback(
    monkeypatch: pytest.MonkeyPatch, counter_state: str,
) -> None:
    if counter_state == "missing":
        monkeypatch.delattr(cache.sys, "getallocatedblocks", raising=False)
    else:
        monkeypatch.setattr(cache.sys, "getallocatedblocks", lambda: 0, raising=False)

    source = "def still_parses(value):\n    return value + 1\n"
    first = cache.parse(source, "no-allocation-counter.py")
    second = cache.parse(source, "no-allocation-counter.py")

    assert isinstance(first, ast.Module)
    assert isinstance(second, ast.Module)
    assert first is second
    assert cache.stats()["trees"] == 1


def test_two_threads_parse_within_bounded_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        "".join(
            f"def worker_{index}(value):\n    result = value + {index + variant}\n    return result\n"
            for index in range(2_000)
        )
        for variant in (0, 1)
    ]
    measured = [_measure_parse(source) for source in sources]
    actuals = [actual for _, actual, _ in measured]
    del measured

    barrier = threading.Barrier(2)
    original_parse = ast.parse

    def synchronized_parse(source: str, *args, **kwargs) -> ast.Module:
        if source in sources:
            barrier.wait(timeout=20)
        return original_parse(source, *args, **kwargs)

    # Both workers capture their process-global block baseline before either parse.
    monkeypatch.setattr(cache.ast, "parse", synchronized_parse)
    with ThreadPoolExecutor(max_workers=2) as pool:
        trees = list(pool.map(lambda row: cache.parse(row[1], f"worker-{row[0]}.py"), enumerate(sources)))

    estimates = [cache._TREES[cache.parse_cache_key(source)][2] for source in sources]
    assert all(isinstance(tree, ast.Module) for tree in trees)
    for actual, estimate in zip(actuals, estimates, strict=True):
        assert actual / max(estimate, 1) <= 1.5
        assert estimate / max(actual, 1) <= 3


def test_estimator_failure_returns_parse_without_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "def estimate_failure(value):\n    return value + 1\n"

    def broken_estimator(
        _source: bytes, _tree: ast.Module, _allocated_blocks: int = 0,
    ) -> int:
        raise RuntimeError("sizing unavailable")

    monkeypatch.setattr(cache, "_estimate_tree_retained_bytes", broken_estimator)
    first = cache.parse(source, "estimate_failure.py")
    second = cache.parse(source, "estimate_failure.py")

    assert isinstance(first, ast.Module)
    assert isinstance(second, ast.Module)
    assert first is not second
    assert cache.stats()["trees"] == 0
    assert cache.stats()["tree_misses"] == 2


def test_tree_cap_stays_at_256_mib() -> None:
    assert cache.MAX_TREE_RETAINED_BYTES == 256 * 1024 * 1024


def test_derived_results_are_keyed_by_source_content() -> None:
    first = cache.derived("source-value", "VALUE = 1\n", "same.py", lambda tree: tree.body[0].value.value)
    second = cache.derived("source-value", "VALUE = 2\n", "same.py", lambda tree: tree.body[0].value.value)

    assert (first, second) == (1, 2)
    assert cache.stats()["derived_misses"] == 2
    assert cache.stats()["derived_hits"] == 0


def test_dense_nodes_and_large_constants_each_affect_estimate() -> None:
    dense = ";".join(f"name_{index} = {index}" for index in range(5_000))
    literal = "VALUE = " + repr("long-value-" * 30_000) + "\n"
    for source in (dense, literal):
        tree, actual, allocated_blocks = _measure_parse(source)
        estimate = cache._estimate_tree_retained_bytes(
            source.encode("utf-8", "surrogatepass"), tree, allocated_blocks,
        )
        assert actual / max(estimate, 1) <= 1.5
        assert estimate / max(actual, 1) <= 3
