"""Exercise cold launch through the real runner, child, and scripted HTTP endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace

import pytest

import kernel_invalidation
from kernel_done_is_submit_authority import candidate_exists, project_final_candidate_tools
from kernel_invalidation import LedgerIndex, runtime_invalidation_index, sync_runtime_invalidation
from kernel_recall import render_runtime
from stateframe import cli, profiles, runner
from test_workspace_state_oracle import scripted_provider


SWITCHES = (
    "STATEFRAME_WORKSPACE_OBJECTS",
    "STATEFRAME_TURN_PANELS",
    "STATEFRAME_TURN_DELIVERY_PROOFS",
    "STATEFRAME_TURN_INDEXED_READS",
    "STATEFRAME_TURN_BOUNDED_QUALIFICATION",
    "STATEFRAME_TURN_GENERATION_OVERLAP",
    "STATEFRAME_TURN_MINIMAL_LAUNCH",
)
TARGET = "group_00/module_00.py"
LAST = "group_59/module_49.py"
PROBE = """
import collections
import json
import sys
import threading
from pathlib import Path

counts = collections.Counter()
before_send = True
builders = {
    "intake_orientation": ("kernel_repo_intel.py", "repo_maps"),
    "build": ("symbol_registry.py", "symbol_scans"),
    "_build_graph_uncached": ("call_graph.py", "call_graphs"),
    "_extract_file": ("call_graph.py", "graph_source_reads"),
    "_scan_repository": ("kernel_coordinated_change_ledger.py", "ledger_scans"),
}

def count(code):
    global before_send
    if not before_send:
        return
    name = code.co_name
    if name == "recorded" and code.co_filename.endswith("/_child.py"):
        endpoint = sys._getframe(2).f_locals.get("endpoint", "")
        if endpoint.rstrip("/").endswith("/chat/completions"):
            before_send = False
            write_counts()
            return
    builder = builders.get(name)
    if builder is not None and code.co_filename.endswith("/" + builder[0]):
        label = builder[1]
        counts[label] += 1
        if counts[label] == 1 or counts[label] % 250 == 0:
            write_counts()

def write_counts():
    Path("startup-work.json").write_text(json.dumps(counts))

if hasattr(sys, "monitoring"):
    monitor = sys.monitoring
    monitor.use_tool_id(5, "cold-start work counts")
    monitor.register_callback(5, monitor.events.PY_START, lambda code, offset: count(code))
    monitor.set_events(5, monitor.events.PY_START)
else:
    def profile(frame, event, arg):
        if event == "call":
            count(frame.f_code)
    sys.setprofile(profile)
    threading.setprofile(profile)

import atexit
atexit.register(write_counts)
"""


def _observe_child(monkeypatch, *, count_work):
    # The integration checkout intentionally leaves release metadata regeneration
    # to packaging. Reuse the checked-in switch set for this workspace benchmark.
    document = json.loads((runner.ENGINE_DIR / "switch-registry.json").read_text())
    registered = tuple(document["switches"])
    assert set(SWITCHES) - {SWITCHES[-1]} <= set(registered)
    monkeypatch.setattr(runner, "engine_switches", lambda *args, **kwargs: frozenset(registered))
    source = (
        "import sys\n"
        f"sys.path.insert(0, {str(runner.ENGINE_DIR)!r})\n"
        "import kernel_release_switch_registry as registry\n"
        f"registry.release_stateframe_switches = lambda root=None: "
        f"(frozenset({registered!r}), {document['digest']!r})\n"
    )
    if count_work:
        source += PROBE
    popen = runner.subprocess.Popen

    def instrument(command, *args, **kwargs):
        if command[-1] == "stateframe._child":
            command = [*command]
            command[3] = source + "\n" + command[3]
        elif len(command) == 4 and str(command[2]).endswith("/_child.py"):
            child, run = command[2:]
            command = [*command[:2], "-c", source + (
                "\nimport runpy\n"
                f"sys.path.insert(0, {str(Path(child).parent)!r})\n"
                f"sys.argv = [{child!r}, {run!r}]\n"
                f"runpy.run_path({child!r}, run_name='__main__')"
            )]
        return popen(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", instrument)


def _run(tmp_path, monkeypatch, *, enabled, large=False):
    repo = tmp_path / "repo"
    if large:
        for group in range(60):
            folder = repo / f"group_{group:02d}"
            folder.mkdir(parents=True)
            for leaf in range(50):
                (folder / f"module_{leaf:02d}.py").write_text(
                    f"def answer():\n    return {group * 50 + leaf}\n"
                )
    else:
        target = repo / TARGET
        target.parent.mkdir(parents=True)
        target.write_text("def answer():\n    return 0\n")

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema": "stateframe-product-profile/1",
        "switches": {
            **profiles.load(profiles.DEFAULT),
            **{name: "on" if name in enabled else "off" for name in SWITCHES},
        },
    }))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.delenv(SWITCHES[-1], raising=False)
    run = tmp_path / "run"
    _observe_child(monkeypatch, count_work=large)
    read_path = LAST if large else TARGET
    turns = 2 if large else 1
    steps = [{"tool": "read", "arguments": {"path": read_path}}] * turns
    with scripted_provider(steps) as (url, trace):
        started = monotonic_ns()
        cli.main([
            "run", "--repo", str(repo), "--task", "Read the implementation of answer.",
            "--edit", TARGET, "--profile", str(profile), "--endpoint", url,
            "--model", "scripted", "--run-dir", str(run), "--quiet",
            "--test-python", sys.executable, "--no-run-env", "--shell", "off",
            "--max-turns", str(turns), "--timeout", "360" if large else "240",
        ])
        elapsed = (trace.requests[0][0] - started) / 1e9 if trace.requests else None
    log = (run / "engine.log").read_text() if (run / "engine.log").exists() else ""
    return run, trace.requests, elapsed, log


@pytest.mark.host_sandbox
def test_cold_launch_reaches_http_without_foreground_indexes(tmp_path, monkeypatch):
    run, requests, elapsed, log = _run(tmp_path, monkeypatch, enabled=SWITCHES, large=True)
    work = run / "startup-work.json"
    counts = json.loads(work.read_text()) if work.is_file() else None
    diagnostic = (f"requests={len(requests)}, first_send_seconds={elapsed}, "
                  f"foreground_work={counts}\n{log[-1800:]}")
    print(diagnostic)
    assert requests and "Unavailable" not in log, diagnostic
    assert counts == {}, diagnostic
    first = json.loads(requests[0][1])
    assert {tool["function"]["name"] for tool in first["tools"]} == {"list_files", "read"}
    assert len(requests) >= 2, diagnostic
    assert LAST.encode() in requests[1][1] and b"return 2999" in requests[1][1], diagnostic
    # Work counts are the deterministic launch budget. Allow loaded hosts time
    # to schedule the child and local HTTP server without hiding a hang.
    assert elapsed < 120, diagnostic


def test_initial_recall_renders_without_building_call_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "answer.py").write_text("def answer():\n    return 1\n")
    receipts = [{"kind": "read", "result_class": "OK", "target_path": "answer.py"}]
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root, _receipts=receipts), receipts=receipts)
    recall = render_runtime(runtime, path="answer.py")
    assert recall
    assert runtime.invalidation_index.propagation_metrics["graph_builds"] == 0

    eager_index = LedgerIndex()
    eager_index.configure_call_graph(root)
    eager_runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root, _receipts=receipts, inval=eager_index),
        receipts=receipts,
    )
    assert eager_index.propagation_metrics["graph_builds"] == 1
    assert render_runtime(eager_runtime, path="answer.py") == recall


def test_call_graph_is_built_on_dependency_demand(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "answer.py").write_text("def answer():\n    return 1\n")
    test_path = "test_answer.py"
    (root / test_path).write_text(
        "from answer import answer\n\ndef test_answer():\n    assert answer() == 1\n",
    )
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root, inval=LedgerIndex()), receipts=[])
    index = runtime_invalidation_index(runtime)
    assert index.dependencies_for("answer.py") == ()
    assert index.propagation_metrics["graph_builds"] == 0

    runtime.receipts.append({"kind": "test", "result_class": "GREEN", "target_path": test_path})
    sync_runtime_invalidation(runtime)
    assert index.observations[0].dependencies == ("answer.py",)
    assert index.dependencies_for(test_path) == ("answer.py",)
    assert index.propagation_metrics["graph_builds"] == 1

    runtime.receipts.append({"kind": "edit", "result_class": "APPLIED", "target_path": "answer.py"})
    sync_runtime_invalidation(runtime)
    assert runtime._invalidation_gate["dirty_reasons"] == ["call_graph_dependency"]
    assert index.propagation_metrics["graph_builds"] == 1


def test_candidate_tool_floor_resumes_after_navigation():
    runtime = SimpleNamespace(
        binding=lambda: {"product_revision": "existing-revision"},
        stateframe_done_is_submit_mode="on",
        minimal_launch_pending=lambda: True,
    )
    tools = [{"function": {"name": "read"}}]
    assert candidate_exists(runtime)
    assert project_final_candidate_tools(runtime, tools, request_id="first") == (tools, ())
    runtime.minimal_launch_pending = lambda: False
    _, missing = project_final_candidate_tools(runtime, tools, request_id="next")
    assert set(missing) >= {"edit", "str_replace", "file_create", "file_delete", "file_move"}


def test_bootstrap_leaves_preflight_receipts_for_the_next_request(tmp_path, monkeypatch):
    receipt_access = []

    def receipts(runtime):
        receipt_access.append(runtime)
        return ()

    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path),
        _minimal_launch=True,
        minimal_launch_pending=lambda: True,
    )
    monkeypatch.setattr(kernel_invalidation, "_runtime_receipts", receipts)
    index = sync_runtime_invalidation(runtime)
    assert receipt_access == []
    runtime.minimal_launch_pending = lambda: False
    assert sync_runtime_invalidation(runtime) is index
    assert receipt_access == [runtime]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("enabled", [
    pytest.param(SWITCHES[:size], id=f"prefix-{size}") for size in range(len(SWITCHES) + 1)
] + [
    pytest.param((SWITCHES[-1],), id="minimal-only"),
    pytest.param(SWITCHES[1:], id="turn-switches-without-workspace-objects"),
    pytest.param((SWITCHES[0], SWITCHES[-1]), id="workspace-objects-and-minimal"),
])
def test_field_switch_combinations_start(tmp_path, monkeypatch, enabled):
    _, requests, _, log = _run(tmp_path, monkeypatch, enabled=enabled)
    assert requests and "Unavailable" not in log and '"result":"ERR:' not in log, log
