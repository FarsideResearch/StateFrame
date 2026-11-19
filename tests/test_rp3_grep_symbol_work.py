"""Work and observation budgets for zero-hit grep definition advice."""
from __future__ import annotations

import ast
from collections import Counter
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import pytest

from kernel_bolt import KernelBolt
from symbol_registry import SymbolRegistry


@pytest.fixture
def switches(monkeypatch):
    from kernel_release_switch_registry import release_stateframe_switches

    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_") and name != "STATEFRAME_HOME":
            monkeypatch.delenv(name)
    # The turn switches are default-off engine arms.
    for key in release_stateframe_switches()[0]:
        if key.startswith("STATEFRAME_TURN_"):
            monkeypatch.setenv(key, "on")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "on")
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_COORDS", "on")
    monkeypatch.setenv("STATEFRAME_REPEAT_GATE", "on")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "scope").mkdir(parents=True)
    (root / "scope" / "notes.txt").write_text("no definitions here\n")
    (root / "scope" / "test_answer.py").write_text("def test_answer(): assert False\n")
    (root / "answer.py").write_text("def answer():\n    return 1\n")
    (root / "first.py").write_text("def shared(a, /, *, key=1):\n    return a\n")
    (root / "second.py").write_text("async def shared(value):\n    return value\n")
    source = "# work-count fixture\n" + "".join(
        f"def filler_{index}(value):\n    return value + {index}\n" for index in range(12))
    for group in range(45):
        directory = root / f"package_{group:02}"
        directory.mkdir()
        for index in range(100):
            if group < 44 or index < 96:
                (directory / f"unit_{index:03}.py").write_text(source)
    return root


@pytest.fixture
def work_counts(monkeypatch):
    counts = Counter()
    thread = threading.get_ident()
    build = SymbolRegistry.build
    parse = ast.parse
    read_text = Path.read_text
    read_bytes = Path.read_bytes

    def counted_build(self, *args, **kwargs):
        counts["registry_builds"] += 1
        return build(self, *args, **kwargs)

    def counted_parse(source, *args, **kwargs):
        if threading.get_ident() == thread and isinstance(source, str) and source.startswith("# work-count fixture"):
            counts["foreground_parses"] += 1
        return parse(source, *args, **kwargs)

    def counted_read(path, *args, **kwargs):
        if threading.get_ident() == thread and path.name.startswith("unit_"):
            counts["foreground_fixture_reads"] += 1
        return read_text(path, *args, **kwargs)

    def counted_bytes(path, *args, **kwargs):
        if threading.get_ident() == thread and path.name.startswith("unit_"):
            counts["foreground_fixture_reads"] += 1
        return read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(SymbolRegistry, "build", counted_build)
    monkeypatch.setattr(ast, "parse", counted_parse)
    monkeypatch.setattr(Path, "read_text", counted_read)
    monkeypatch.setattr(Path, "read_bytes", counted_bytes)
    return counts


@pytest.fixture
def indexed_work(monkeypatch, work_counts, repo):
    import kernel_bolt

    # The same regression file runs on the base, where the work seam is absent.
    view = getattr(kernel_bolt, "_GrepSymbolView", None)
    if view is None:
        return None
    ready = threading.Event()
    release = threading.Event()
    lookup = view.lookup
    rows = []
    for name in ("first.py", "second.py"):
        rows.extend(SymbolRegistry.definitions_in_source(name, (repo / name).read_bytes())["shared"])

    def counted_build(root, oracle, publish, current):
        # Control when posted facts become ready so cold and warm work budgets
        # are independent of background parsing speed. Other cases build real facts.
        try:
            if release.wait(30) and current():
                publish(view({"shared": tuple(rows)}, None, current))
        finally:
            ready.set()

    def counted_lookup(self, symbol):
        work_counts["admitted_lookups"] += 1
        return lookup(self, symbol)

    monkeypatch.setattr(kernel_bolt, "_build_grep_symbol_view", counted_build)
    monkeypatch.setattr(view, "lookup", counted_lookup)
    return release, ready


def _check_budget(label, samples):
    print(f"{label} per action: {samples}")
    unexpected = {"registry_builds", "foreground_parses", "foreground_fixture_reads"}
    assert all(not unexpected.intersection(counts) for _, counts in samples), samples
    assert max(seconds for seconds, _ in samples) < 10, samples


def test_zero_hit_bolt_callback_budget(repo, switches, work_counts, indexed_work):
    from kernel_navigation_pages import search_repository

    bolt = KernelBolt(repo, "answer", "grep-callback", defer_symbols=True)
    samples = []
    outcomes = []
    for index, pattern in enumerate(("shared", "shared", "shared", "missing")):
        page = search_repository(repo, pattern, "scope", mode="literal")
        assert page.metadata["total"] == 0 and page.metadata["scan_complete"]
        bolt.last_grep_hits = 0
        work_counts.clear()
        started = time.monotonic()
        outcomes.append(bolt.observe(f"grep {pattern} scope", page.text))
        samples.append((time.monotonic() - started, dict(work_counts)))
        if indexed_work is not None and index == 0 and "index pending" in outcomes[-1]:
            indexed_work[0].set()
            assert indexed_work[1].wait(30)
    _check_budget("bolt callback", samples)
    assert [counts.get("admitted_lookups", 0) for _, counts in samples] == [0, 1, 0, 1]
    assert "index pending" in outcomes[0]
    assert "EXISTS (2 definitions)" in outcomes[1]
    assert "EXISTS (2 definitions)" in outcomes[2]
    assert "EXISTS" not in outcomes[3]


class _StopAfterGrep(BaseException):
    pass


def test_zero_hit_native_grep_budget(repo, tmp_path, switches, work_counts, indexed_work,
                                     monkeypatch):
    import kernel_navigation_pages
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    # Run intake in the small seed. The measured grep actions see the large
    # fixture, which is moved into the sandbox before the first search begins.
    seed = tmp_path / "seed"
    seed.mkdir()
    shutil.copy2(repo / "answer.py", seed / "answer.py")
    shutil.copytree(repo / "scope", seed / "scope")
    copy = create(seed, tmp_path / "copy")
    row = build_row(copy, task="Inspect the answer", target="answer.py", task_id="grep-work")
    output = tmp_path / "run" / "worker" / "measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("REACT_NATIVE_TOOL_BATCHES", "1")
    monkeypatch.setenv("STATEFRAME_VIEW", "operational")
    monkeypatch.setenv("STATEFRAME_PROJECTOR", "rich")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 7)
    queries = [{"pattern": pattern, "path": "scope", "mode": "literal"}
               for pattern in ("shared", "shared", "shared", "missing")]
    queries[2]["file_glob"] = "*.py"
    greps = [{"tool": "grep", "arguments": query} for query in queries]
    actions = [{"tool": "read", "required": True, "arguments": {"path": "answer.py"}},
               {**greps[0], "required": True, "also": greps[1:]}]
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": actions})
    server = serve(script)
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    ask = react_repair.ask_tools
    observe = KernelBolt.observe
    search = kernel_navigation_pages.search_repository
    samples = []
    outcomes = []
    owners = []
    started = 0.0

    def capture_action(*args, **kwargs):
        if len(samples) == len(queries):
            raise _StopAfterGrep
        return ask(*args, **kwargs)

    def capture_search(root, pattern, scope=None, **kwargs):
        nonlocal started
        if scope and Path(scope).name == "scope":
            if not started:
                for path in repo.iterdir():
                    if not (root / path.name).exists():
                        path.rename(root / path.name)
            work_counts.clear()
            started = time.monotonic()
        return search(root, pattern, scope, **kwargs)

    def capture_observation(self, command, observation, **kwargs):
        result = observe(self, command, observation, **kwargs)
        if command.startswith("grep "):
            assert started > 0
            samples.append((time.monotonic() - started, dict(work_counts)))
            outcomes.append(result)
            assert (self.root / "package_44" / "unit_095.py").is_file()
            owners.append(self)
            if indexed_work is not None and len(samples) == 1 and "index pending" in result:
                indexed_work[0].set()
                assert indexed_work[1].wait(30)
            if len(samples) == len(queries):
                raise _StopAfterGrep
        return result

    capture_action.minimal_launch = True
    monkeypatch.setattr(react_repair, "ask_tools", capture_action)
    monkeypatch.setattr(KernelBolt, "observe", capture_observation)
    monkeypatch.setattr(kernel_navigation_pages, "search_repository", capture_search)
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [context.set(None) for context in contexts]
    try:
        with pytest.raises(_StopAfterGrep):
            react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for context, token in zip(contexts, tokens):
            context.reset(token)

    script.assert_consumed()
    assert len(owners) == len(queries)
    grep_receipts = [receipt for receipt in owners[0]._receipts if receipt["kind"] == "grep"]
    assert [receipt["result_class"] for receipt in grep_receipts] == ["NO_HITS"] * len(queries)
    assert len({receipt["receipt_id"] for receipt in grep_receipts}) == len(queries)
    assert all(receipt["result_set_digest"] for receipt in grep_receipts)
    _check_budget("native grep", samples)
    assert [counts.get("admitted_lookups", 0) for _, counts in samples] == [0, 1, 0, 1]
    assert "index pending" in outcomes[0]
    assert "EXISTS (2 definitions)" in outcomes[1]
    assert "EXISTS (2 definitions)" in outcomes[2]
    assert "EXISTS" not in outcomes[3]
    assert all("scan_complete=true" in result for result in outcomes)


def _scoped_observation(root, pattern):
    from kernel_navigation_pages import search_repository

    return search_repository(root, pattern, "scope", mode="literal").text


def _observe(bolt, pattern):
    bolt.last_grep_hits = 0
    return bolt.observe(f"grep {pattern} scope", _scoped_observation(bolt.root, pattern))


def _ready_observation(bolt, pattern):
    deadline = time.monotonic() + 10
    while True:
        observation = _observe(bolt, pattern)
        if "index pending" not in observation:
            return observation
        assert time.monotonic() < deadline, "Python facts did not become ready"
        time.sleep(0.01)


def test_legacy_symbol_registry_builds_once_on_demand(tmp_path, switches, work_counts, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "off")
    (tmp_path / "scope").mkdir()
    source = tmp_path / "answer.py"
    source.write_text("def answer(value, /): pass\n\nasync def second(): pass\n")

    bolt = KernelBolt(tmp_path, "", "lookup")
    assert bolt.observe("list_files", "answer.py") == "answer.py"
    assert bolt.observe("read answer.py", source.read_text()) == source.read_text()
    assert bolt.observe("grep answer", "answer.py:1:def answer") == "answer.py:1:def answer"
    assert work_counts["registry_builds"] == 0

    first = _observe(bolt, "answer")
    assert "answer.py:1" in first
    assert "def answer(value, /)" in first
    assert work_counts["registry_builds"] == 1
    assert "answer.py:3" in _observe(bolt, "second")
    assert work_counts["registry_builds"] == 1


@pytest.mark.parametrize("oracle", ["off", "on"])
def test_completed_facts_match_legacy_and_refresh_after_edit_and_delete(
        tmp_path, switches, oracle, monkeypatch):
    root = tmp_path / "repo"
    (root / "scope").mkdir(parents=True)
    first = root / "first.py"
    second = root / "second.py"
    first.write_text("def lone(): pass\n\ndef shared(a, /, *, option=False): pass\n")
    second.write_text("async def shared(value): pass\n")
    monkeypatch.setenv("STATEFRAME_EXISTENCE_ORACLE", oracle)
    with monkeypatch.context() as off:
        off.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
        off.setenv("STATEFRAME_TURN_INDEXED_READS", "off")
        legacy = KernelBolt(root, "", "legacy")
        expected = {name: _observe(legacy, name) for name in ("lone", "shared", "share")}
    bolt = KernelBolt(root, "", "indexed")
    assert [_ready_observation(bolt, name) for name in expected] == list(expected.values())
    assert expected["share"] == _scoped_observation(root, "share")

    first.write_text("\n\n\ndef shared(updated): pass\n")
    second.unlink()
    bolt.receipt("edit", "replace", "first.py", "APPLIED", changed_paths=("first.py", "second.py"))
    updated = _ready_observation(bolt, "shared")
    assert "first.py:4" in updated
    assert "first.py:4" not in expected["shared"]
    assert "second.py:" not in updated
    assert _ready_observation(bolt, "lone") == _scoped_observation(root, "lone")


def test_unavailable_index_and_partial_grep_cannot_assert_absence(tmp_path, switches, monkeypatch):
    from mc_repo_intel_vendor import background

    root = tmp_path / "repo"
    (root / "scope").mkdir(parents=True)
    (root / "answer.py").write_text("def answer(): pass\n")
    requests = []

    def corrupt(root, kind, work):
        requests.append((root, kind))
        return object()

    monkeypatch.setattr(background, "request", corrupt)
    bolt = KernelBolt(root, "", "unavailable")
    partial = "[search page] total=0 complete=true scan_complete=false\nscan incomplete"
    bolt.last_grep_hits = 0
    assert bolt.observe("grep pending scope", partial) == partial
    assert requests == []
    pending = _observe(bolt, "answer")
    assert pending.startswith(_scoped_observation(root, "answer"))
    assert "index pending" in pending
    assert "EXISTS" not in pending
    assert len(requests) == 1
