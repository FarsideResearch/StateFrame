"""Smoke work budgets and freshness at the request and completion boundaries."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Event, current_thread
from time import perf_counter
from types import SimpleNamespace

import kernel_ship_tree_smoke as smoke
import pytest
from kernel_exec_session import CleanupReport, ExecResult
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState


@pytest.fixture
def enabled(monkeypatch):
    switches = (
        "WORKSPACE_OBJECTS", "TURN_PANELS", "TURN_DELIVERY_PROOFS", "TURN_INDEXED_READS",
        "TURN_BOUNDED_QUALIFICATION", "TURN_GENERATION_OVERLAP", "TURN_MINIMAL_LAUNCH",
        "TURN_ATOMIC_PUBLICATION", "TURN_REPO_INTEL", "TURN_TIMING", "SHIP_TREE_SMOKE",
    )
    for name in switches:
        monkeypatch.setenv(f"STATEFRAME_{name}", "on")
    monkeypatch.setenv("PYTHONPATH", "")


def runtime_for(tmp_path, count):
    root = tmp_path / "project"
    root.mkdir()
    paths = tuple(f"module_{index:04}.py" for index in range(count))
    for index, path in enumerate(paths):
        (root / path).write_text(f"import absent_provider\nVALUE = {index}\n", encoding="utf-8")
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    runtime = object.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, current_step=0, task_row={"files": paths})
    events = []
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    runtime.stateframe_ship_tree_smoke_mode = "on"
    runtime._minimal_launch = True
    return runtime, consumers, paths, events


def completed(paths, *, error=None, cleanup=True):
    rows = [{"path": path, "status": "FAIL" if error else "PASS", "error": error}
            for path in paths]
    return ExecResult(0, False, b"\nSTATEFRAME_SMOKE_RESULT=" + json.dumps(rows).encode(),
                      False, 0, 0, CleanupReport(cleanup, 0, (), "probe cleanup"))


def instrument(monkeypatch, root, *, own_thread=None):
    counts = Counter()

    def track(name, original, *, path=False):
        def counted(*args, **kwargs):
            if ((own_thread is None or current_thread() is own_thread)
                    and (not path or args[0].is_relative_to(root))):
                counts[name] += 1
            return original(*args, **kwargs)
        return counted

    monkeypatch.setattr(smoke, "_source_roots", track("root_builds", smoke._source_roots))
    monkeypatch.setattr(smoke, "local_import_shadows",
                        track("shadow_parses", smoke.local_import_shadows))
    monkeypatch.setattr(Path, "read_text", track("text_reads", Path.read_text, path=True))
    monkeypatch.setattr(Path, "read_bytes", track("byte_reads", Path.read_bytes, path=True))
    return counts, track


def test_minimal_navigation_never_joins_blocked_smoke(tmp_path, monkeypatch, enabled):
    runtime, consumers, _, events = runtime_for(tmp_path, 4_500)
    counts, track = instrument(monkeypatch, runtime.owner.root, own_thread=current_thread())
    release = Event()

    def blocked(_root, paths, _roots, _timeout):
        release.wait(3)
        return completed(paths)

    monkeypatch.setattr(smoke, "_execute_smoke", track("interpreter_runs", blocked))
    try:
        start = perf_counter()
        messages = runtime._apply_rulebook_mechanisms([], provider_request=True)
        elapsed = perf_counter() - start
        print(f"minimal navigation {elapsed:.6f}s, foreground work {dict(counts)}")
        assert not counts
        assert consumers.state.work.content_reads == 0
        assert elapsed < 10
        assert "PENDING" in messages[0]["content"]
        assert not events and getattr(runtime, "_ship_tree_smoke_result", None) is None
    finally:
        release.set()


def test_unchanged_and_single_blob_smoke_work(tmp_path, monkeypatch, enabled):
    runtime, consumers, paths, _ = runtime_for(tmp_path, 4_500)
    runtime.owner.current_step = 1
    counts, track = instrument(monkeypatch, runtime.owner.root)
    monkeypatch.setattr(smoke, "_execute_smoke",
                        track("interpreter_runs", lambda _root, paths, _roots, _limit: completed(paths)))
    start = perf_counter()
    initial = runtime.ship_tree_smoke()
    print(f"valid baseline {perf_counter() - start:.6f}s, work {dict(counts)}")
    assert initial.status == "PASS"

    counts.clear()
    start = perf_counter()
    for _ in range(2):
        assert runtime.ship_tree_smoke() == initial
        assert consumers.state.work.content_reads == 0
    elapsed = perf_counter() - start
    print(f"two warm requests {elapsed:.6f}s, work {dict(counts)}")

    warm_counts = counts.copy()
    counts.clear()
    changed_path = paths[0]
    runtime._ship_tree_touched = [changed_path]
    with consumers.state.begin_write((changed_path,)):
        (runtime.owner.root / changed_path).write_text("VALUE = -1\n", encoding="utf-8")
    start = perf_counter()
    changed = runtime.ship_tree_smoke()
    print(f"single edit {perf_counter() - start:.6f}s, work {dict(counts)}")
    assert changed.status == "PASS" and changed.tree_digest != initial.tree_digest
    assert not warm_counts
    assert elapsed < 2
    assert counts["shadow_parses"] <= 1
    assert counts["text_reads"] + counts["byte_reads"] <= 1
    assert counts["interpreter_runs"] == 1


def test_provider_config_link_and_interpreter_changes_require_fresh_probe(
        tmp_path, monkeypatch, enabled):
    runtime, consumers, paths, _ = runtime_for(tmp_path, 1)
    root = runtime.owner.root
    runtime.owner.current_step = 1
    calls = []
    cleanup_ok = True

    def execute(_root, checked, roots, _timeout):
        calls.append(roots)
        return completed(checked, cleanup=cleanup_ok)

    monkeypatch.setattr(smoke, "_execute_smoke", execute)
    monkeypatch.setenv("STATEFRAME_TEST_PYTHON", str(tmp_path / "python"))
    for version in ("one", "two"):
        interpreter = tmp_path / version
        interpreter.write_text(version)
        interpreter.chmod(0o700)
    selected = tmp_path / "python"
    selected.symlink_to("one")
    assert runtime.ship_tree_smoke().roots == (str(root),)
    assert runtime.ship_tree_smoke().checked == paths and len(calls) == 1

    provider = root / "sibling"
    with consumers.state.begin_write(("sibling",)):
        provider.mkdir()
        (provider / "absent_provider.py").touch()
    assert runtime.ship_tree_smoke().roots == (str(root), str(provider))
    assert len(calls) == 2

    config_root = root / "configured"
    with consumers.state.begin_write(("pytest.ini", "configured")):
        config_root.mkdir()
        (root / "pytest.ini").write_text("[pytest]\npythonpath = configured\n")
    assert runtime.ship_tree_smoke().roots == (str(root), str(config_root), str(provider))
    assert len(calls) == 3

    # A linked provider's referent can change with an unchanged workspace tree.
    external = tmp_path / "external.py"
    external.touch()
    with consumers.state.begin_write(("sibling/absent_provider.py",)):
        linked = provider / "absent_provider.py"
        linked.unlink()
        linked.symlink_to(external)
    assert runtime.ship_tree_smoke().roots[-1] == str(provider)
    external.unlink()
    assert runtime.ship_tree_smoke().roots == (str(root), str(config_root))
    assert len(calls) == 5
    with consumers.state.begin_write(("sibling/absent_provider.py",)):
        linked.unlink()
    assert runtime.ship_tree_smoke().status == "PASS"
    assert runtime.ship_tree_smoke().status == "PASS" and len(calls) == 6

    selected.unlink()
    selected.symlink_to("two")
    cleanup_ok = False
    with pytest.raises(smoke.ImportSmokeCleanupError):
        runtime.ship_tree_done_gate()
    cleanup_ok = True
    assert runtime.ship_tree_done_gate() == (True, "")
    assert runtime.ship_tree_done_gate() == (True, "") and len(calls) == 8


@pytest.mark.parametrize("change", ["scope", "parent", "live"])
def test_bootstrap_probe_is_published_only_for_its_original_basis(
        tmp_path, monkeypatch, enabled, change):
    runtime, consumers, paths, events = runtime_for(tmp_path, 2)
    entered, release = Event(), Event()
    calls = []

    def execute(_root, checked, _roots, _timeout):
        calls.append(checked)
        if len(calls) == 1:
            entered.set()
            release.wait(5)
            return completed(checked, error="ImportError: old evidence")
        return completed(checked)

    monkeypatch.setattr(smoke, "_execute_smoke", execute)
    try:
        assert "PENDING" in runtime._apply_rulebook_mechanisms([], provider_request=True)[0]["content"]
        assert entered.wait(5)
        if change == "scope":
            runtime.owner.task_row["files"] = paths[1:]
        elif change == "parent":
            runtime.owner.task_row["parent_revision"] = "replacement"
        else:
            with consumers.state.begin_write((paths[0],)):
                (runtime.owner.root / paths[0]).write_text("VALUE = 10\n")
        runtime.owner.current_step = 1
        release.set()
        assert runtime.ship_tree_done_gate() == (True, "")
        assert len(calls) == 2
        assert runtime._ship_tree_smoke_result.status == "PASS"
        assert all(not fields.get("result", {}).get("failures") for _, fields in events)
    finally:
        release.set()


def test_background_cleanup_failure_reaches_completion_gate(tmp_path, monkeypatch, enabled):
    runtime, consumers, _, _ = runtime_for(tmp_path, 1)
    monkeypatch.setattr(smoke, "_execute_smoke",
                        lambda _root, paths, _roots, _timeout: completed(paths, cleanup=False))
    assert "PENDING" in runtime._apply_rulebook_mechanisms([], provider_request=True)[0]["content"]
    runtime.owner.current_step = 1
    with pytest.raises(smoke.ImportSmokeCleanupError):
        runtime.ship_tree_done_gate()
    assert runtime._ship_tree_smoke_result is None
    assert consumers.state.fence.root == runtime.owner.root


def test_large_smoke_scope_keeps_argv_and_result_bounded(tmp_path, monkeypatch, enabled):
    root = tmp_path / "project"
    root.mkdir()
    checked = tuple(f"{'directory' * 10}/module_{index}.py" for index in range(4_500))
    observed = {}

    class Session:
        def __init__(self, _directory):
            pass

        def run(self, argv, capability):
            assert argv[-2] == "--inputs"
            observed["input"] = Path(argv[-1])
            assert json.loads(observed["input"].read_text()) == [list(checked), [str(root)]]
            assert 256 << 10 < capability.limits.output_bytes <= 4 << 20
            return completed(checked)

        def close(self):
            pass

    monkeypatch.setattr("kernel_exec_session.ExecSession", Session)
    assert smoke._execute_smoke(root, checked, (str(root),), 20).exit_code == 0
    assert not observed["input"].exists()


def test_ready_smoke_keeps_legacy_sidecar_and_verdict(tmp_path, monkeypatch, enabled):
    runtime, consumers, _, _ = runtime_for(tmp_path, 1)
    runtime.owner.current_step = 1
    monkeypatch.setattr(
        smoke, "_execute_smoke",
        lambda _root, paths, _roots, _timeout: completed(paths, error="ImportError: parent"),
    )
    observed = []
    for switch in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_TURN_DELIVERY_PROOFS", switch)
        runtime._ship_tree_baseline = None
        runtime._ship_tree_smoke_result = None
        messages = runtime._apply_rulebook_mechanisms([], provider_request=True)
        observed.append((json.dumps(messages).encode(), runtime._ship_tree_smoke_result.as_dict(),
                         runtime.ship_tree_done_gate()))
    assert observed[0] == observed[1]
    assert observed[1][1]["status"] == "ENVIRONMENT"
    assert consumers.state.fence.root == runtime.owner.root
