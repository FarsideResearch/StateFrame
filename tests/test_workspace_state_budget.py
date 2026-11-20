"""Measure work at filesystem and provider boundaries on a large read-only turn."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager

import pytest
from _journal import measurement_events
from test_workspace_state_oracle import scripted_provider

from stateframe import runner
from stateframe.endpoint import Endpoint
from stateframe.engine.kernel_tool_settings import ToolSettings

BULK_FILES = 4_998  # Two additional Python files reach the workcopy limit of 5,000.
SEND_ONE_LIMIT = 60
TURN_LIMIT = 10
POINT_STAT_LIMIT = 64
RUN_BACKSTOP = 240


class WorkspaceBudgetExceeded(AssertionError):
    """A completed provider trajectory exceeded a workspace budget."""


def _install_io_probe(roots, output, replace=setattr):
    """Run in the engine child, before loading the engine or opening the workspace."""
    import builtins
    import io
    from functools import wraps

    events = []
    owner_pid = os.getpid()

    def observe(operation, kind):
        @wraps(operation)
        def counted(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.fsdecode(path)
                directory = kwargs.get("dir_fd")
                if directory is not None and not os.path.isabs(name):
                    try:
                        name = os.path.join(os.readlink(f"/proc/self/fd/{directory}"), name)
                    except OSError:
                        return operation(path, *args, **kwargs)
                name = os.path.abspath(name)
                for root in roots:
                    if name == root or name.startswith(root + os.sep):
                        if kind == "walk" or name.endswith(".py"):
                            events.append((time.monotonic_ns(), kind, name[len(root):]))
                        break
            return operation(path, *args, **kwargs)

        return counted

    def save():
        if os.getpid() == owner_pid:
            with open(output, "w", encoding="utf-8") as stream:
                json.dump(events, stream)

    for module, kind, names in (
        (os, "walk", ("scandir", "listdir")),
        (os, "stat", ("stat", "lstat")),
        (os, "open", ("open",)),
        (io, "open", ("open",)),
        (builtins, "open", ("open",)),
    ):
        for name in names:
            replace(module, name, observe(getattr(module, name), kind))
    return save


@contextmanager
def _measure_child_io(monkeypatch, repo, run, output):
    original = subprocess.Popen
    # Add the test probe to the production bootstrap before runpy enters the child.
    # Keep the original process identity, working directory and environment.
    source = "import atexit, json, os, time\n" + inspect.getsource(_install_io_probe)
    roots = tuple(str(path) for path in (run / "workspace", run / "parent", repo))
    source += f"\natexit.register(_install_io_probe({roots!r}, {str(output)!r}))"

    def launch(command, *args, **kwargs):
        if len(command) == 5 and command[1:3] == ["-B", "-c"] and command[4] == "stateframe._child":
            command = list(command)
            command[3] = command[3].replace("runpy.run_path(", f"exec({source!r});runpy.run_path(", 1)
        elif len(command) == 4 and command[2] == str(runner.CHILD):
            python, _, child, run_path = command
            bootstrap = (
                "import os, runpy, sys;"
                f"sys.path[0]={str(runner.CHILD.parent)!r};sys.argv={[child, run_path]!r};"
                f"exec({source!r});runpy.run_path({child!r},run_name='__main__')"
            )
            command = [python, "-B", "-c", bootstrap, child]
        return original(command, *args, **kwargs)

    # Admission runs in the CLI parent. Count it independently of the engine child.
    with monkeypatch.context() as patch:
        patch.setattr(runner.subprocess, "Popen", launch)
        save = _install_io_probe(
            roots, str(output.with_name("parent-io.json")), patch.setattr,
        )
        yield
    save()


def _large_repo(repo):
    for index in range(BULK_FILES):
        group, leaf = divmod(index, 50)
        directory = repo / "modules" / f"group_{group:03d}"
        if leaf == 0:
            directory.mkdir(parents=True)
        (directory / f"entry_{leaf:02d}.py").write_text(f"VALUE = {index}\n")
    (repo / "answer.py").write_text("VALUE = 1\n")
    (repo / "test_answer.py").write_text("from answer import VALUE\n\ndef test_value():\n    assert VALUE == 2\n")


def _work(events, start, end):
    counts = Counter()
    stat_paths = set()
    for timestamp, kind, path in events:
        if start <= timestamp < end:
            counts[kind] += 1
            if path.startswith("/modules/") and kind == "stat":
                stat_paths.add(path)
            if path.startswith("/modules/") and kind == "open":
                counts["bulk_open"] += 1
    return {**dict.fromkeys(("walk", "stat", "open", "bulk_open"), 0),
            **counts, "bulk_stat_paths": len(stat_paths)}


@pytest.mark.host_sandbox
@pytest.mark.xfail(
    strict=True, raises=WorkspaceBudgetExceeded,
    reason="Build orders 6 and 10 remove legacy prepare scans; order 11 enforces launch/send budgets",
)
def test_large_workspace_unchanged_turn_budget(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _large_repo(repo)
    run = tmp_path / "run"
    probe = tmp_path / "engine-io.json"
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    read = {"tool": "read", "arguments": {"path": "answer.py"}}
    # The second read observes the same bytes. Two completed responses provide
    # two unchanged response-to-request intervals, including send qualification.
    with (
        scripted_provider([read, read, read]) as (url, trace),
        _measure_child_io(monkeypatch, repo, run, probe),
    ):
        options = runner.RunOptions(
            repo=repo, task="Read answer.py and fix the failing check.",
            endpoint=Endpoint.from_args(url, "scripted", None), run_dir=run,
            target="answer.py", test_python=sys.executable, max_turns=3,
            tool_settings=ToolSettings(run_env=False), timeout_s=RUN_BACKSTOP, quiet=True,
        )
        start = time.monotonic_ns()
        try:
            summary = runner.run(options)
        except runner.RunDeadlineExceeded as error:
            elapsed = (time.monotonic_ns() - start) / 1e9
            first_send = ((trace.requests[0][0] - start) / 1e9 if trace.requests else None)
            raise WorkspaceBudgetExceeded(
                f"{len(trace.requests)} requests by the {RUN_BACKSTOP}s run backstop "
                f"(observed {elapsed:.3f}s; send 1={first_send}; limit {SEND_ONE_LIMIT}s)"
            ) from error
    assert summary["engine_result"] == "RESPONSE_BUDGET_EXHAUSTED", summary
    requests = [stamp for stamp, _ in trace.requests[:3]]
    responses = trace.responses[:3]
    trace.script.assert_consumed()
    assert len(requests) == len(responses) == 3
    events = (json.loads(probe.read_text())
              + json.loads(probe.with_name("parent-io.json").read_text()))
    launch = (requests[0] - start) / 1e9
    gaps = [(request - response) / 1e9 for request, response in zip(requests[1:], responses)]
    launch_work = _work(events, start, requests[0])
    turns = [_work(events, response, request)
             for request, response in zip(requests[1:], responses)]
    prepare = [row["host_seconds"] for row in measurement_events(run)
               if row.get("kind") == "request_prepare_timing"]
    assert len(prepare) >= len(requests), prepare
    prepare = prepare[:len(requests)]
    print(f"launch={launch:.3f}s {launch_work}; gaps={gaps}; prepare={prepare}; turns={turns}")

    missed = []
    if launch >= SEND_ONE_LIMIT:
        missed.append(f"launch {launch:.3f}s >= {SEND_ONE_LIMIT}s")
    for index, (gap, elapsed, work) in enumerate(zip(gaps, prepare[1:], turns), start=2):
        if gap >= TURN_LIMIT or elapsed >= TURN_LIMIT:
            missed.append(f"send {index}: gap={gap:.3f}s prepare={elapsed:.3f}s (limit {TURN_LIMIT}s)")
        if work["walk"] or work["bulk_open"] or work["bulk_stat_paths"] > POINT_STAT_LIMIT:
            missed.append(f"send {index}: unchanged workspace I/O {work}")
    if missed:
        raise WorkspaceBudgetExceeded("\n".join(missed))


@pytest.mark.xfail(
    strict=True, raises=WorkspaceBudgetExceeded,
    reason="Build order 6 reuses the validated editable projection on an unchanged workspace",
)
def test_unchanged_broad_editable_projection_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kernel_request_source_window import editable_product_paths

    repo = tmp_path / "repo"
    _large_repo(repo)
    paths = [path.relative_to(repo).as_posix() for path in sorted((repo / "modules").rglob("*.py"))]
    assert len(paths) == BULK_FILES
    runtime = SimpleNamespace(owner=SimpleNamespace(
        root=repo, target="answer.py", task_row={"editable_paths": paths},
    ))
    first = editable_product_paths(runtime)
    assert first and first[0] == "modules/group_000/entry_00.py"
    output = tmp_path / "io.json"
    with monkeypatch.context() as patch:
        save = _install_io_probe((str(repo),), str(output), patch.setattr)
        start = time.monotonic_ns()
        second = editable_product_paths(runtime)
        end = time.monotonic_ns()
    save()
    assert second == first
    work = _work(json.loads(output.read_text()), start, end)
    elapsed = (end - start) / 1e9
    print(f"unchanged editable projection={elapsed:.3f}s {work}")
    if (elapsed >= TURN_LIMIT or work["walk"] or work["bulk_open"]
            or work["bulk_stat_paths"] > POINT_STAT_LIMIT):
        raise WorkspaceBudgetExceeded(f"unchanged editable projection={elapsed:.3f}s {work}")


def test_prepare_entry_detects_a_new_workspace_walk(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import kernel_stateframe_runtime as engine

    class PrepareEntryObserved(Exception):
        pass

    def stop_at_first_host_action(*args):
        raise PrepareEntryObserved

    repo = tmp_path / "repo"
    _large_repo(repo)
    runtime = engine.StateFrameRuntime.__new__(engine.StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=repo, current_step=2)
    output = tmp_path / "entry-io.json"
    # This control isolates the prepare prologue from the known legacy scans.
    # The full worker test above measures the rest of the boundary through send.
    with monkeypatch.context() as patch:
        patch.setattr(engine._host_authorship, "open_step", stop_at_first_host_action)
        save = _install_io_probe((str(repo),), str(output), patch.setattr)
        start = time.monotonic_ns()
        with pytest.raises(PrepareEntryObserved):
            runtime.prepare([], uncapped=[], tools=[])
        end = time.monotonic_ns()
    save()
    work = _work(json.loads(output.read_text()), start, end)
    elapsed = (end - start) / 1e9
    print(f"prepare entry={elapsed:.3f}s {work}")
    assert (work["walk"], work["bulk_open"], work["bulk_stat_paths"]) == (0, 0, 0), work
