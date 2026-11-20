"""Regression gates for scoped turn fact reads."""

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import pytest

from kernel_invalidation import LedgerIndex
from repo_intel_core.call_graph import CallSite, CodeGraph, SymbolDef


class CountingDefinitions(dict):
    reads = 0
    comparisons = 0

    def __getitem__(self, key):
        self.reads += 1
        return super().__getitem__(key)

    def items(self):
        for row in super().items():
            self.comparisons += 1
            yield row


def _add_definition(graph, path, name):
    symbol = f"{path}::{name}"
    graph._index_def(SymbolDef(symbol, path, name, name, 1, 2, "function", False))
    return symbol


def _large_graph(root: Path):
    graph = CodeGraph()
    graph.defs = CountingDefinitions()
    for index in range(4800):
        path = f"provider_{index:04}.py"
        (root / path).write_text("def compute():\n    return 1\n", encoding="utf-8")
        _add_definition(graph, path, "compute")

    calls = []
    for index in range(200):
        path = f"test_query_{index:03}.py"
        (root / path).write_text(
            "from provider_0000 import compute as answer\n"
            "def test_query():\n    assert answer() == 1\n",
            encoding="utf-8",
        )
        symbol = _add_definition(graph, path, "test_query")
        calls.append((path, CallSite(symbol, "answer", 0, False, (), False, None, 3)))
        graph._add_edge(symbol, "provider_0000.py::compute")
    return graph, calls


def test_point_query_work_independent_of_unrelated_files(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "on")
    graph, calls = _large_graph(tmp_path)
    ledger = LedgerIndex()
    ledger.configure_call_graph(graph)
    imports = {path: {"answer": "provider_0000.py::compute"} for path, _ in calls}
    graph.defs.reads = graph.defs.comparisons = 0

    start = perf_counter()
    for path, call in calls:
        assert [symbol for symbol, _ in graph._resolve_callee(call, imports)] == [
            "provider_0000.py::compute",
        ]
        assert ledger.dependencies_for(path) == ("provider_0000.py",)
    seconds = perf_counter() - start

    counts = {"definition_reads": graph.defs.reads,
              "unrelated_comparisons": graph.defs.comparisons}
    print(f"5,000 files, 200 scoped queries: {seconds:.6f}s {counts}")
    assert counts["unrelated_comparisons"] == 0, (counts, seconds)
    assert counts["definition_reads"] <= 3 * len(calls), (counts, seconds)
    assert seconds < 0.40, (counts, seconds)


class RetainedWorkspace:
    def __init__(self, snapshots):
        from kernel_turn_contracts import Ref

        self.snapshots = snapshots
        self.admission = Ref("turn/admission/1", "0" * 64, 0)
        self.reads = self.walks = self.diffs = 0

    @contextmanager
    def retain(self, root):
        from kernel_turn_contracts import Snapshot

        assert root in self.snapshots
        yield Snapshot(root, self.admission, 0)

    def changes(self, old, new):
        self.diffs += 1
        before, after = self.snapshots[old.root], self.snapshots[new.root]
        return iter(sorted(path for path in before.keys() | after.keys()
                           if before.get(path) != after.get(path)))

    def read(self, at, path):
        self.reads += 1
        try:
            return self.snapshots[at.root][path]
        except KeyError:
            raise FileNotFoundError(os.fsdecode(path)) from None

    def materialize(self, parent, destination):
        self.walks += 1
        destination.mkdir()
        for relative, data in self.snapshots[parent.root].items():
            path = destination / os.fsdecode(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


class PublicationArena:
    def __init__(self):
        self.objects = {}

    def stage(self, codec, data):
        from kernel_turn_contracts import Ref

        ref = Ref(codec, sha256(data).hexdigest(), len(data))
        self.objects[ref] = data
        return ref

    def rows(self, page):
        return json.loads(self.objects[page.data.value])


def _basis(live="parent"):
    from kernel_turn_contracts import Basis, Head

    return Basis("parent", live, 0, Head(0, "genesis"), 0, "authority")


def _open_facts(directory, workspace, arena, *, attempt="attempt"):
    from kernel_turn_contracts import Context
    from kernel_turn_facts import open_port

    return open_port(Context("principal", "run", attempt, directory, "profile"),
                     workspace=workspace, publication=arena)


def _warm_process(directory):
    from unittest.mock import patch

    from kernel_turn_contracts import Key
    from kernel_turn_fact_store import FactStore
    from symbol_registry import SymbolRegistry

    workspace = RetainedWorkspace({"parent": {}})
    arena = PublicationArena()
    facts = _open_facts(directory, workspace, arena)
    visited = 0
    stored_rows = FactStore.rows

    def count_rows(*args):
        nonlocal visited
        for row in stored_rows(*args):
            visited += 1
            yield row

    with patch.object(FactStore, "rows", side_effect=count_rows), patch.object(
            SymbolRegistry, "definitions_in_source", side_effect=AssertionError("warm parse")):
        start = perf_counter()
        for index in range(200):
            name = f"query_{index:03}"
            page = facts.lookup(_basis(), Key("symbol", name.encode()), None, 1)
            assert arena.rows(page)[0]["file"] == f"query_{index:03}.py"
            assert page.next_cursor is None
        seconds = perf_counter() - start
    counts = (visited, workspace.reads, workspace.walks, workspace.diffs)
    print(f"Warm fresh process, 200 queries: {seconds:.6f}s {counts}", flush=True)
    assert counts == (200, 0, 0, 0), counts
    assert seconds < 0.40, seconds


def test_persisted_queries_read_only_their_postings(tmp_path, monkeypatch):
    pytest.importorskip("kernel_turn_facts")
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "on")
    files = {f"provider_{index:04}.py".encode(): b"def compute(): pass\n"
             for index in range(4800)}
    files.update({f"query_{index:03}.py".encode(): f"def query_{index:03}(): pass\n".encode()
                  for index in range(200)})
    workspace = RetainedWorkspace({"parent": files})
    arena = PublicationArena()
    facts = _open_facts(tmp_path, workspace, arena)
    start = perf_counter()
    with workspace.retain("parent") as parent:
        facts.build(parent, lambda: False)
    print(f"Cold fact ingestion: {perf_counter() - start:.6f}s "
          f"({workspace.reads} source reads, {workspace.walks} materializations)")
    assert (workspace.reads, workspace.walks) == (5000, 1)

    run = subprocess.run(
        [sys.executable, "-c",
         "from pathlib import Path; import runpy, sys; "
         "runpy.run_path(sys.argv[1])['_warm_process'](Path(sys.argv[2]))",
         __file__, str(tmp_path)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    print(run.stdout, end="")
    assert run.returncode == 0, run.stdout + run.stderr


def test_changed_provider_order_and_absence_match_fresh_projection(tmp_path, monkeypatch):
    pytest.importorskip("kernel_turn_facts")
    from kernel_turn_contracts import Key, Stale, Unavailable
    from symbol_registry import SymbolRegistry

    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "on")
    parent_files = {b"b.py": b"def answer(): pass\n", b"z.py": b"def answer(): pass\n"}
    live_files = {b"a.py": b"def answer(): pass\ndef new_name(): pass\n",
                  b"b.py": parent_files[b"b.py"]}
    workspace = RetainedWorkspace({"parent": parent_files, "live": live_files})
    arena = PublicationArena()
    facts = _open_facts(tmp_path, workspace, arena)
    parent_basis, live_basis = _basis(), _basis("live")
    symbol = Key("symbol", b"answer")
    missing = Key("symbol", b"new_name")
    point = Key("workspace-path", b"a.py")

    # First demand can read admitted bytes while optional ingestion is pending.
    cold = facts.lookup(live_basis, point, None, 1)
    assert arena.rows(cold)[0]["blob"]["sha256"] == sha256(live_files[b"a.py"]).hexdigest()
    assert workspace.walks == 0
    with pytest.raises(Unavailable, match="incomplete"):
        facts.lookup(parent_basis, missing, None, 1)
    with workspace.retain("parent") as parent:
        facts.build(parent, lambda: False)

    first = facts.lookup(parent_basis, symbol, None, 1)
    assert arena.rows(first)[0]["file"] == "b.py"
    assert arena.rows(facts.lookup(parent_basis, symbol, first.next_cursor, 1))[0]["file"] == "z.py"
    with pytest.raises(Stale):
        facts.lookup(live_basis, symbol, first.next_cursor, 1)
    absent = facts.lookup(parent_basis, missing, None, 1)
    assert arena.rows(absent) == []
    assert facts.current(parent_basis, absent.data.deps)
    assert not facts.current(live_basis, absent.data.deps)
    assert not facts.current(live_basis, first.data.deps)
    assert not facts.current(parent_basis, cold.data.deps)

    oracle = tmp_path / "oracle"
    with workspace.retain("live") as live:
        workspace.materialize(live, oracle)
    expected = sorted(SymbolRegistry().build(oracle).where("answer"), key=lambda row: row["file"])
    projected = facts.lookup(live_basis, symbol, None, 16)
    assert arena.rows(projected) == expected
    assert workspace.diffs == 1

    # Reopening the persisted parent index in another attempt gives identical
    # dependencies and artifact bytes, including deletion and shadowing.
    resumed = _open_facts(tmp_path, workspace, arena, attempt="resumed")
    assert resumed.lookup(live_basis, symbol, None, 16) == projected
    assert resumed.lookup(live_basis, missing, None, 16).data.deps != absent.data.deps


def test_cancelled_build_and_damaged_index_never_certify_absence(tmp_path, monkeypatch):
    pytest.importorskip("kernel_turn_facts")
    import sqlite3

    from kernel_turn_contracts import Corrupt, Key, Unavailable

    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "on")
    workspace = RetainedWorkspace({"parent": {
        b"a.py": b"def answer(): pass\n", b"b.py": b"def other(): pass\n",
    }})
    arena = PublicationArena()
    facts = _open_facts(tmp_path, workspace, arena)
    key = Key("symbol", b"answer")
    with workspace.retain("parent") as parent:
        facts.build(parent, lambda: workspace.reads > 0)
        assert workspace.reads == 1
        with pytest.raises(Unavailable, match="incomplete"):
            facts.lookup(_basis(), key, None, 8)
        facts.build(parent, lambda: False)
    assert len(arena.rows(facts.lookup(_basis(), key, None, 8))) == 1
    store, = tmp_path.glob("turn-facts-*/symbols.sqlite")
    with sqlite3.connect(store) as db:
        db.execute("DELETE FROM symbols WHERE name = ?", ("answer",))
    with pytest.raises(Corrupt, match="membership"):
        facts.lookup(_basis(), key, None, 8)


def test_menu_paths_stay_ordered_and_skip_unrelated_file_probes(tmp_path, monkeypatch):
    pytest.importorskip("kernel_turn_facts")
    from kernel_request_source_window import editable_product_paths

    paths = [f"file_{index:04}.py" for index in range(50)]
    for path in paths[1:]:
        (tmp_path / path).touch()
    target = tmp_path / paths[-1]
    runtime = SimpleNamespace(target=target, owner=SimpleNamespace(
        root=tmp_path, task_row={"editable_paths": list(reversed(paths)) + paths}))
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "off")
    expected = editable_product_paths(runtime)
    assert expected[0] == target.name
    assert paths[0] not in expected
    calls = 0
    is_file = Path.is_file

    def count_file(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        return is_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", count_file)
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "on")
    assert editable_product_paths(runtime) == expected
    assert calls == len(expected) + 1
