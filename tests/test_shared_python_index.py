"""Measure the three send-one consumers with cold, warm and edited source."""
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _measure(root):
    import kernel_coordinated_change_ledger as ledger
    from repo_intel_core import call_graph
    from symbol_registry import SymbolRegistry

    counts = {"parses": 0, "module_walks": 0}
    parse = ast.parse
    children = ast.iter_child_nodes

    def counted_parse(source, *args, **kwargs):
        if isinstance(source, str) and source.startswith("# indexed fixture"):
            counts["parses"] += 1
        return parse(source, *args, **kwargs)

    def counted_children(node):
        if isinstance(node, ast.Module):
            counts["module_walks"] += 1
        return children(node)

    ast.parse = counted_parse
    ast.iter_child_nodes = counted_children
    visit = ast.NodeVisitor.visit

    def counted_visit(visitor, node):
        if isinstance(node, ast.Module):
            counts["module_walks"] += 1
        return visit(visitor, node)

    ast.NodeVisitor.visit = counted_visit
    started = time.perf_counter()
    registry = SymbolRegistry().build(root)
    scan = ledger.scan_repository(root)
    graph = call_graph.build_graph(str(root))
    assert registry.files_scanned == len(scan.definitions) == len(graph.defs)
    assert {s.symbol for s in scan.definitions} == set(registry.defs)
    assert {s.short_name for s in graph.defs.values()} == set(registry.defs)
    print(json.dumps({**counts, "seconds": round(time.perf_counter() - started, 3)}))


def test_send_one_python_index_work(tmp_path):
    count = 4500
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(count):
        (root / f"m{index}.py").write_text(
            f"# indexed fixture\ndef f{index}(x):\n    return x + {index}\n")
    runs = []
    for attempt in range(3):
        if attempt == 2:
            (root / "m0.py").write_text("# indexed fixture\ndef changed(x):\n    return x\n")
        # Each worker has its own graph cache and process-local parse caches.
        env = dict(os.environ, HOME=str(tmp_path), STATEFRAME_HOME=str(tmp_path / "state"),
                   STATEFRAME_REPO_CACHE_DIR=str(tmp_path / f"worker-{attempt}"))
        reply = subprocess.run([sys.executable, __file__, str(root)], env=env,
                               text=True, capture_output=True, check=True, timeout=180)
        runs.append(json.loads(reply.stdout))
    print("cold / warm / one edit", runs)
    assert [r["parses"] for r in runs] == [count, 0, 1], runs
    assert [r["module_walks"] for r in runs] == [count, 0, 1], runs
    assert max(r["seconds"] for r in runs) < 60, runs


if __name__ == "__main__":
    _measure(Path(sys.argv[1]))


def test_nested_source_matches_original_index_bytes(tmp_path, monkeypatch):
    import hashlib

    import kernel_coordinated_change_ledger as ledger
    import kernel_parse_cache as cache
    from repo_intel_core import call_graph
    from symbol_registry import SymbolRegistry

    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "state"))
    source = """from target import task as run
import target as alias
K = alias
class Service:
    VALUE = 3
    @staticmethod
    async def method(x, /, *args, flag=True, **kwargs):
        from target import other as run
        def nested(y):
            if y:
                return nested(y - 1)
            return 0
        for a in args:
            for b in a:
                run(b)
        return nested(x)
action = lambda x: run(x)
"""
    (root / "target.py").write_text("def task(x): return x\nother = task\n")
    path = root / "subject.py"
    path.write_text(source)
    tree = cache.parse(source, path.name)
    before = ast.dump(tree, include_attributes=True)
    for _ in range(2):
        registry = SymbolRegistry().build(root)
        scan = ledger._scan_repository(root)
        extract = call_graph._extract_file(path, root)
        assert call_graph._build_import_table(tree, root, root) == extract.import_table
        encoded = json.dumps([registry.defs, scan.to_dict(), extract.to_dict()],
                             sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == "171d9d1b445f5d94562ad3daf4c367d150065735d764a9517fc3ded0d509d246"
        assert ast.dump(tree, include_attributes=True) == before
        extract.defs[0].properties["modified"] = True
        extract.import_table.clear()
        registry.defs.clear()
        cache.clear()


def test_graph_imports_follow_sibling_creation(tmp_path, monkeypatch):
    from repo_intel_core import call_graph

    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "graph"))
    (root / "consumer.py").write_text("import provider as p\ndef work(): return p.task()\n")
    (root / "alternative.py").write_text("def task(): pass\n")
    call_graph.build_graph(str(root))
    (root / "provider.py").write_text("def task(): pass\n")
    graph = call_graph.build_graph(str(root))
    assert [ref.file for ref in graph.find_callees("consumer.py::work")] == ["provider.py"]


def test_invalid_persisted_rows_rebuild_from_current_source(tmp_path, monkeypatch):
    import hashlib
    import sqlite3

    import kernel_parse_cache as cache
    from symbol_registry import SymbolRegistry

    root = tmp_path / "repository"
    root.mkdir()
    (root / "one.py").write_text("def actual(): pass\n")
    home = tmp_path / "state"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    assert SymbolRegistry().build(root).exists("actual")
    (database,) = (home / "python-index").glob("*.sqlite")
    cache.clear()
    with sqlite3.connect(database) as db:
        key, = db.execute("SELECT key FROM files").fetchone()
        payload = b'[["malformed"]]'
        checksum = hashlib.sha256(key + b"registry" + payload).digest()
        db.execute("UPDATE files SET registry=?", (checksum + payload,))
    registry = SymbolRegistry().build(root)
    assert registry.where("actual")[0]["line"] == 1


def test_unavailable_cache_keeps_overlay_and_registry_independent(tmp_path, monkeypatch):
    import kernel_coordinated_change_ledger as ledger
    from symbol_registry import SymbolRegistry

    root = tmp_path / "repository"
    root.mkdir()
    (root / "one.py").write_text("def actual(): pass\n")
    monkeypatch.setenv("STATEFRAME_HOME", str(root))
    scan = ledger.scan_repository(root, overlays={"one.py": "def pending(): pass\n"})
    assert [site.symbol for site in scan.definitions] == ["pending"]
    assert list(SymbolRegistry().build(root).defs) == ["actual"]
    assert list(root.iterdir()) == [root / "one.py"]


@pytest.mark.parametrize("threads", [True, False])
def test_registry_read_ahead_preserves_order_errors_and_long_sources(tmp_path, monkeypatch, threads):
    from concurrent.futures import ThreadPoolExecutor

    from symbol_registry import SymbolRegistry

    if not threads:
        def unavailable(*args, **kwargs):
            raise RuntimeError("threads unavailable")

        monkeypatch.setattr(ThreadPoolExecutor, "submit", unavailable)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "state"))
    root = tmp_path / "repository"
    root.mkdir()
    for i in range(75):
        (root / f"m{i}.py").write_bytes(b"\xff" if i == 48 else b"def shared(): pass\r\n")
    (root / "large.py").write_text("#" + "x" * 100000 + "\ndef tail(): pass\n")
    expected = [p.name for p in root.rglob("*.py") if p.name not in ("m48.py", "large.py")]
    registry = SymbolRegistry().build(root)
    assert [row["file"] for row in registry.where("shared")] == expected
    assert registry.where("tail")[0]["line"] == 2
    assert (registry.files_scanned, registry.files_failed) == (75, 1)


def test_older_graph_cache_upgrades_without_claiming_a_source_edit(tmp_path, monkeypatch):
    from repo_intel_core import call_graph

    root = tmp_path / "repo"
    root.mkdir()
    (root / "example.py").write_text("def value(): return 1\n")
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "graph"))
    call_graph.build_graph(str(root))
    path = call_graph.repo_cache_file(root)
    data = json.loads(path.read_text())
    data["files"]["example.py"].pop("python_imports", None)
    path.write_text(json.dumps(data))
    graph = call_graph.build_graph(str(root))
    assert graph.rebuilt_files == set()
    assert "example.py::value" in graph.defs


def test_uncached_graph_keeps_persistence_disabled(tmp_path, monkeypatch):
    from repo_intel_core import call_graph

    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("def value(): pass\n")
    home = tmp_path / "state"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "graph"))
    assert "source.py::value" in call_graph.build_graph(str(root), use_cache=False).defs
    assert not home.exists()
