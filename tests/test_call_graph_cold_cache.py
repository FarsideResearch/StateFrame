import ast
import pickle

import kernel_parse_cache
from kernel_invalidation import LedgerIndex
from repo_intel_core import call_graph, lsp_resolve


def _memoized_graph(root):
    return call_graph.build_graph(str(root), use_cache=False, use_memo=True)


def test_python_file_facts_share_ast_and_survive_a_cold_build(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "attempt-cache"))
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source = "from impl import target as alias\r\n\r\ndef run(): return alias()\r\n"
    for root in (first, second):
        (root / "caller.py").write_bytes(source.encode())

    parses = []
    original_parse = ast.parse

    def record(source, filename="<unknown>", *args, **kwargs):
        if filename == "caller.py" or str(filename).endswith(("/caller.py", "/impl.py")):
            parses.append(filename)
        return original_parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", record)
    kernel_parse_cache.clear()
    kernel_parse_cache.derived("test", source.replace("\r\n", "\n"), "caller.py",
                               lambda tree: len(tree.body))
    initial = _memoized_graph(first)
    assert initial.find_definition("run").file == "caller.py"
    assert initial.find_callees("run") == []
    assert parses == ["caller.py"]

    kernel_parse_cache.clear()
    call_graph.clear_build_memo()
    (second / "impl.py").write_text("def target(): return 1\n", encoding="utf-8")
    cached = _memoized_graph(second)
    assert [ref.symbol for ref in cached.find_callees("run")] == ["impl.py::target"]
    assert parses == ["caller.py", str(second / "impl.py")]

    (second / "caller.py").write_text("def changed(): return 2\n", encoding="utf-8")
    revised = _memoized_graph(second)
    assert revised.find_definition("run") is None
    assert revised.find_definition("changed").file == "caller.py"
    assert parses[-1] == str(second / "caller.py") and len(parses) == 3


def test_ledger_build_defers_resolution_until_dependencies_are_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "attempt-cache"))
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("CODE_GRAPH_LSP", "on")
    monkeypatch.setattr(lsp_resolve, "_JEDI_AVAILABLE", True)
    root = tmp_path / "checkout"
    (root / "tests").mkdir(parents=True)
    (root / "impl.py").write_text("def implementation(): return 1\n", encoding="utf-8")
    (root / "facade.py").write_text("from impl import implementation as exposed\n", encoding="utf-8")
    test_file = "tests/test_feature.py"
    (root / test_file).write_text(
        "from facade import exposed\n\ndef test_feature(): return exposed()\n", encoding="utf-8",
    )

    finalized = []
    consulted = []
    original_finalize = call_graph.CodeGraph._finalize

    def finalize(self, *args, **kwargs):
        finalized.append(self)
        return original_finalize(self, *args, **kwargs)

    def resolve(repo_root, sites_by_file, **kwargs):
        consulted.append(repo_root)
        site, = sites_by_file[test_file]
        assert site.name == "exposed"
        target = lsp_resolve.ResolvedTarget("impl.py", 1, "implementation", "function")
        return {test_file: {site.key: target}}, {}

    monkeypatch.setattr(call_graph.CodeGraph, "_finalize", finalize)
    monkeypatch.setattr(lsp_resolve, "resolve_repo", resolve)
    ledger = LedgerIndex()
    graph = ledger.configure_call_graph(root)
    assert graph.find_definition("test_feature").file == test_file
    assert graph.coverage_summary()["indexed"] == 3
    assert finalized == consulted == []

    restored = pickle.loads(pickle.dumps(ledger))
    assert restored.dependencies_for(test_file) == ("impl.py",)
    assert restored.dependencies_for(test_file) == ("impl.py",)
    resolved, = finalized
    caller = f"{test_file}::test_feature"
    assert resolved.edge_strategy(caller, "impl.py::implementation") == call_graph.STRATEGY_LSP
    assert resolved.lsp_stats["overridden"] == 1
    assert consulted == [root]
