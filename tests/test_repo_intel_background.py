"""Work budgets for repository intelligence on the engine's foreground paths."""

import importlib
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

from mc_repo_intel_vendor import call_graph, lsp_resolve, repo_map, ts_extract


def test_broken_grammar_is_probed_once(tmp_path, monkeypatch, record_property):
    attempts = []

    def get_parser(language):
        attempts.append(language)
        raise LookupError(language)

    pack = ModuleType("tree_sitter_language_pack")
    pack.get_parser = get_parser
    monkeypatch.setitem(sys.modules, pack.__name__, pack)
    monkeypatch.setattr(ts_extract, "_tslp", pack)
    monkeypatch.setattr(ts_extract, "_TSLP_IMPORTED", True)
    monkeypatch.setattr(ts_extract, "_tslp_functional", None)
    isolated_map = ModuleType(repo_map.__name__)
    isolated_map.__package__ = repo_map.__package__
    loader = importlib.util.spec_from_file_location(repo_map.__name__, repo_map.__file__).loader
    loader.exec_module(isolated_map)
    isolated_map._MAX_FILES = 301

    for index in range(5000):
        suffix = ".ts" if index < 300 else ".txt"
        (tmp_path / f"file_{index:04d}{suffix}").write_text(
            "export function work() {}\n", encoding="utf-8")

    start = time.monotonic()
    result, stats = isolated_map._generate_repo_map_with_stats(str(tmp_path))
    elapsed = time.monotonic() - start
    record_property("map_seconds", round(elapsed, 6))
    record_property("parser_attempts", len(attempts))
    assert stats["text_scan"] == 300 and "file_0299.ts" in result
    assert len(attempts) == 1, f"{len(attempts)} parser attempts in {elapsed:.3f}s"


def _repo(root):
    root.mkdir()
    for index in range(20):
        (root / f"file_{index:02d}.py").write_text(
            "def run(): return 1\n\ndef call(service): return service.run()\n",
            encoding="utf-8")
    return root


def _enable(monkeypatch, cache_root):
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "on")
    monkeypatch.setenv("REPO_INTEL_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(cache_root / "legacy"))
    monkeypatch.setenv("CODE_GRAPH_LSP", "on")
    monkeypatch.setattr(lsp_resolve, "_JEDI_AVAILABLE", True)


def _graph(root):
    return call_graph.build_graph(str(root), use_cache=False, use_memo=True)


def _ready(root):
    deadline = time.monotonic() + 20
    while True:
        graph = _graph(root)
        if getattr(graph, "index_status", "ready") == "ready":
            return graph
        assert time.monotonic() < deadline, "background index failed to finish"
        time.sleep(0.01)


def test_slow_resolver_stays_off_foreground(tmp_path, monkeypatch, record_property):
    from kernel_invalidation import LedgerIndex

    root = _repo(tmp_path / "workspace")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("LSP_RESOLVE_TOTAL_BUDGET_SECONDS", "0.5")
    calls = []
    started = threading.Event()

    def resolve(*args, **kwargs):
        calls.append((args[1], threading.current_thread().name))
        started.set()
        time.sleep(5)
        return {}

    monkeypatch.setattr(lsp_resolve, "resolve_file_full", resolve)
    ledger = LedgerIndex()
    before = time.monotonic()
    ledger.configure_call_graph(root)
    elapsed = time.monotonic() - before
    record_property("foreground_seconds", round(elapsed, 6))
    record_property("foreground_resolver_calls", sum(name == "MainThread" for _, name in calls))
    assert elapsed < 1, f"foreground graph request took {elapsed:.3f}s, {len(calls)} resolver calls"
    assert started.wait(10), "without a background resolver the wall budget is vacuous"
    assert len(calls) == 1
    assert getattr(_graph(root), "index_status", "ready") != "ready"


def test_runtime_refreshes_the_background_graph_after_dependency_demand(tmp_path, monkeypatch):
    from kernel_invalidation import runtime_invalidation_index

    root = _repo(tmp_path / "workspace")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
    started = threading.Event()
    release = threading.Event()
    original = call_graph._build_graph_uncached

    def build(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        started.set()
        assert release.wait(10), "graph worker was not released"
        return original(*args, **kwargs)

    monkeypatch.setattr(call_graph, "_build_graph_uncached", build)
    try:
        ledger = runtime_invalidation_index(runtime)
        assert ledger._call_graph is None
        assert not started.is_set()
        assert ledger.dependencies_for("tests/test_service.py") == ()
        assert ledger._call_graph.index_status == "pending"
        assert started.wait(10), "dependency lookup did not schedule the graph worker"
    finally:
        release.set()

    ready = _ready(root)
    assert runtime_invalidation_index(runtime) is ledger
    assert ledger._call_graph is ready


def test_warm_run_resolves_only_changed_blobs(tmp_path, monkeypatch, record_property):
    first = _repo(tmp_path / "first")
    second = _repo(tmp_path / "second")
    third = _repo(tmp_path / "third")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("LSP_RESOLVE_TOTAL_BUDGET_SECONDS", "30")
    calls = []

    def resolve(root, path, sites, **kwargs):
        calls.append(path)
        return {site.key: None for site in sites}

    monkeypatch.setattr(lsp_resolve, "resolve_file_full", resolve)
    before = time.monotonic()
    expected = _ready(first).coverage_summary()
    cold_elapsed = time.monotonic() - before
    assert len(calls) == 20
    calls.clear()
    before = time.monotonic()
    assert _ready(second).coverage_summary() == expected
    warm_elapsed = time.monotonic() - before
    record_property("cold_seconds", round(cold_elapsed, 6))
    record_property("warm_seconds", round(warm_elapsed, 6))
    record_property("warm_resolver_calls", len(calls))
    assert not calls, (
        f"warm run re-resolved {len(calls)} unchanged files "
        f"(cold {cold_elapsed:.3f}s, warm {warm_elapsed:.3f}s)")
    (third / "file_08.py").write_text(
        "def run(): return 2\n\ndef call(service): return service.run()\n",
        encoding="utf-8")
    assert _ready(third).coverage_summary() == expected
    record_property("changed_resolver_calls", len(calls))
    assert calls == ["file_08.py"]


def test_mutation_fences_pre_edit_graph(tmp_path, monkeypatch):
    from kernel_live_projection import WorkspaceViewStore
    from kernel_repo_intel import host_where_defined

    root = _repo(tmp_path / "workspace")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    assert _ready(root).find_definition("run") is not None
    path = root / "file_08.py"
    path.write_text("def renamed(): return 10\n", encoding="utf-8")
    WorkspaceViewStore(root, "fixture").note_mutation((path.name,), "applied")

    assert host_where_defined(root, "renamed")["status"] == "PARTIAL"
    assert _ready(root).find_definition("renamed").file == path.name


def test_request_keeps_its_graph_until_a_workspace_mutation(tmp_path, monkeypatch):
    from kernel_live_projection import WorkspaceViewStore
    from kernel_repo_dependencies import DependencyView

    root = _repo(tmp_path / "workspace")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    test = root / "test_service.py"
    test.write_text("from file_00 import run\n\ndef test_service(): run()\n")
    view = DependencyView(lambda: None)
    view.advance(root)
    _ready(root)

    assert view.fact(root, test.name, 0) == (("file_00.py",), False)
    view.advance(root)
    assert view.fact(root, test.name, 0) == (("file_00.py",), True)

    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    test.write_text("from file_01 import run\n\ndef test_service(): run()\n")
    WorkspaceViewStore(root, "fixture").note_mutation((test.name,), "applied")

    assert view.fact(root, test.name, 1) == (("file_01.py",), False)


def test_intake_does_not_join_the_map_worker(tmp_path, monkeypatch):
    from kernel_repo_intel import intake_orientation

    root = _repo(tmp_path / "workspace")
    _enable(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    original = repo_map._generate_repo_map_with_stats
    release = threading.Event()
    entered = threading.Event()
    threads = []

    def generate(*args, **kwargs):
        threads.append(threading.current_thread().name)
        entered.set()
        if threading.current_thread() is not threading.main_thread():
            assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(repo_map, "_generate_repo_map_with_stats", generate)
    try:
        orientation = intake_orientation(root)
        assert "Repository signatures pending." in orientation
        assert entered.wait(10)
        assert threads == ["repo-index-map"]
    finally:
        release.set()

    deadline = time.monotonic() + 20
    while "Repository signatures pending." in (orientation := intake_orientation(root)):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert "file_08.py:" in orientation
