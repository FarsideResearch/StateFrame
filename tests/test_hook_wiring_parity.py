"""Raw no-hooks parity against this candidate's immediate base release.

SF_TEST_BASELINE_IMMEDIATE may name an explicit checkout. Otherwise use the in-tree merge base with
origin/main when it is a strict ancestor; without that ref, use the newest ancestor release commit
(excluding HEAD) whose subject identifies a StateFrame version or release. Both sides explicitly
disable the stable tool head: this parity is about hooks, not tool lists.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from acceptance._support import assert_tree_imports, tree_python

import kernel_measurement as measurement
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from tools.pinned_tree import pinned_tree
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified


CURRENT = Path(__file__).resolve().parents[1]
_RELEASE_SUBJECT = re.compile(
    r"^(?:StateFrame .+: version and changelog|Release v?\d+(?:\.\d+)+|"
    r"Merge pull request #\d+ from .*/release/v?\d+(?:\.\d+)+)$"
)
RUNNER_SPEC_FIELDS = {"check_cmd", "check_source", "protect_tests",
                      "protected_test_globs", "runner_kind"}
pytestmark = pytest.mark.host_sandbox


def _git_text(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(CURRENT), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _immediate_base_release() -> str:
    head = _git_text("rev-parse", "--verify", "HEAD^{commit}")
    if head is None:
        pytest.skip("cannot derive the immediate base release because HEAD is unavailable")

    if _git_text("rev-parse", "--verify", "origin/main^{commit}") is not None:
        merge_base = _git_text("merge-base", "HEAD", "origin/main")
        if merge_base is not None and merge_base != head:
            return merge_base

    history = _git_text("log", "--format=%H%x09%s", "HEAD")
    if history is not None:
        for line in history.splitlines():
            commit, separator, subject = line.partition("\t")
            if separator and commit != head and _RELEASE_SUBJECT.fullmatch(subject):
                return commit
    pytest.skip("cannot find an ancestor release commit to use as the hook parity baseline")


@pytest.fixture(scope="module")
def baseline_checkout(tmp_path_factory):
    explicit = os.environ.get("SF_TEST_BASELINE_IMMEDIATE")
    if explicit is not None:
        if not explicit.strip():
            pytest.fail("SF_TEST_BASELINE_IMMEDIATE is set but empty", pytrace=False)
        baseline = Path(explicit).expanduser().resolve()
    else:
        baseline = pinned_tree(_immediate_base_release(), tmp_path_factory)
    if not (baseline / "src/stateframe/runner.py").is_file():
        pytest.fail(f"immediate base checkout has no src/stateframe/runner.py: {baseline}", pytrace=False)
    return baseline


def _write_launchers(launcher: Path, wrapper: Path) -> None:
    launcher.write_text(
        "import secrets, subprocess, sys\n"
        "from stateframe import cli\n"
        "wrapper, trace, *args = sys.argv[1:]\n"
        "original_hex = secrets.token_hex\n"
        "used = False\n"
        "def fixed_hex(size=None):\n"
        "    global used\n"
        "    if size == 16 and not used:\n"
        "        used = True\n"
        "        return 'a' * 32\n"
        "    return original_hex(size)\n"
        "secrets.token_hex = fixed_hex\n"
        "original_popen = subprocess.Popen\n"
        "def traced_popen(command, *positional, **keywords):\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        command = [command[0], command[1], wrapper, command[2], command[3], trace]\n"
        "    return original_popen(command, *positional, **keywords)\n"
        "subprocess.Popen = traced_popen\n"
        "raise SystemExit(cli.main(args))\n"
    )
    wrapper.write_text(
        "import json, os, runpy, sys\n"
        "child, run, trace = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.argv = [child, run]\n"
        "try:\n"
        "    runpy.run_path(child, run_name='__main__')\n"
        "finally:\n"
        "    with open(trace, 'w') as target:\n"
        "        json.dump(sorted(sys.modules), target)\n"
    )


def _run_checkout(
    checkout: Path, repo: Path, run: Path, launcher: Path, wrapper: Path, trace: Path,
    *, stable_tool_head: str | None,
):
    python = tree_python(checkout / "src",
                         run.parent / f"{run.name}-{checkout.name}-python")
    assert_tree_imports(python, checkout / "src", run.parent)
    script = Script(
        json.loads((CURRENT / "examples/discount_bug.script.json").read_text())
    )
    server = serve(script)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    try:
        command = [
            python,
            str(launcher),
            str(wrapper),
            str(trace),
            "run",
            "--repo",
            str(repo),
            "--task",
            "Fix the pricing code so the tests pass.",
            "--endpoint",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--model",
            "scripted",
            "--run-dir",
            str(run),
            "--test-python",
            sys.executable,
            "--max-turns",
            "16",
            "--quiet",
        ]
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=240
        )
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        requests = [
            path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
        ]
        assert requests
        events = []
        for path in measurement.journal_paths(run / "worker" / "measurement-runs"):
            with measurement.open_journal(path) as stream:
                events.extend(json.loads(line) for line in stream if line.strip())
        kinds = [event["kind"] for event in events]
        # Hash-chain fields and tokenizer measurement/endpoint identity vary
        # between runs because each scripted server uses a fresh port.
        diet_payloads = [
            {key: value for key, value in event.items()
             if key not in {"timestamp_ns", "previous_hash", "record_hash",
                            "run_id", "task_id", "event_id",
                            "tokenize_latency_ms", "tokenize_route",
                            "tokenizer_identity"}}
            for event in events if event["kind"] == "request_diet_applied"
        ]
        tape = (
            (run / "tool-tape.jsonl").read_bytes()
            if (run / "tool-tape.jsonl").exists()
            else None
        )
        return (
            requests,
            kinds,
            tape,
            set(json.loads((run / "spec.json").read_text())),
            json.loads(trace.read_text()),
            diet_payloads,
        )
    finally:
        server.shutdown()
        server.server_close()


def test_no_hooks_raw_requests_and_journal_match_release(tmp_path, baseline_checkout):
    assert (baseline_checkout / "src/stateframe/runner.py").is_file()
    repo = tmp_path / "repo"
    source = CURRENT / "examples/discount_bug"
    import shutil

    shutil.copytree(source, repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(
        baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline_modules.json", stable_tool_head="off",
    )
    run.rename(tmp_path / "baseline_run")
    current = _run_checkout(
        CURRENT, repo, run, launcher, wrapper, tmp_path / "current_modules.json",
        stable_tool_head="off",
    )
    assert current[:3] == baseline[:3]
    assert current[3] == baseline[3] | RUNNER_SPEC_FIELDS
    assert current[5], "fixture did not exercise request-diet journal payloads"
    assert current[5] == baseline[5]
    assert "hooks" not in current[3]
    assert not {
        "kernel_hooks",
        "kernel_hook_exec",
        "kernel_hook_driver",
        "kernel_hook_wiring",
    } & set(current[4])
    assert not any(
        (run / name).exists() for name in ("hooks", "hook-bundle", "hook-tape.jsonl")
    )


def test_no_hooks_regular_stateframe_file_matches_release(tmp_path, baseline_checkout):
    repo = tmp_path / "repo"
    import shutil

    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    (repo / ".stateframe").write_text("ordinary repository data", encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline.json",
                             stable_tool_head="off")
    run.rename(tmp_path / "baseline_run")
    current = _run_checkout(CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
                            stable_tool_head="off")
    assert current[:3] == baseline[:3]
    assert current[3] == baseline[3] | RUNNER_SPEC_FIELDS
    assert current[5] == baseline[5]


def _resume_checkout(checkout: Path, repo: Path, run: Path, launcher: Path,
                     wrapper: Path, trace: Path, *,
                     stable_tool_head: str | None) -> tuple[list[bytes], list[bytes]]:
    python = tree_python(checkout / "src",
                         run.parent / f"{run.name}-{checkout.name}-python")
    assert_tree_imports(python, checkout / "src", run.parent)
    class InterruptAfterTwoActions(Script):
        def __init__(self):
            super().__init__({"model": "scripted", "steps": [
                {"tool": "read", "required": True, "arguments": {"path": "shop/pricing.py"}},
                {"tool": "str_replace", "required": True, "arguments": {
                    "path": "shop/pricing.py",
                    "old_str": "    return round(price * percent / 100, 2)",
                    "new_str": "    return round(price * (100 - percent) / 100, 2)",
                }},
                {"tool": "run_tests", "required": True,
                 "arguments": {"target": "tests/test_pricing.py"}},
                {"tool": "done", "required": True, "arguments": {"summary": "Checks finished."}},
            ], "tokenize": {"mode": "ok"}})
            self.waiting = threading.Event()
            self.release = threading.Event()

        def next_call(self, offered):
            if self.position == 2 and not self.release.is_set():
                self.waiting.set()
                if not self.release.wait(60):
                    raise TimeoutError("interrupted request was not released")
                raise ScriptStepUnavailable("request interrupted before a response")
            return super().next_call(offered)

    script = InterruptAfterTwoActions()
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    command = [python, str(launcher), str(wrapper), str(trace), "run",
               "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
               "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
               "--test-python", sys.executable, "--max-turns", "16", "--quiet"]
    process = None
    try:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        assert script.waiting.wait(600), "first attempt did not reach the third request"
        original = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
                    if path.with_name(path.name.replace(".request.", ".response.")).exists()]
        assert len(original) == 2
        children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
        assert len(children) == 1, children
        child = int(children[0])
        identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        signal_run_verified(run, child, identity, signal.SIGKILL,
                            deadline=time.monotonic() + 120, what="hook wiring parity")
        script.release.set()
        process.communicate(timeout=90)
        resumed = subprocess.run([python, "-m", "stateframe", "resume", str(run),
                                  "--endpoint", endpoint], env=environment, capture_output=True,
                                 text=True, timeout=240)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        rebuilt = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert rebuilt[:2] == original
        return original, rebuilt
    finally:
        script.release.set()
        if process is not None and process.poll() is None:
            # Failure cleanup may precede the first checkpoint.
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="hook parity cleanup", require_checkpoint_saved=False)
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()


def test_no_hooks_resume_rebuilt_requests_match_release(tmp_path, baseline_checkout):
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _resume_checkout(baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline.json",
                                stable_tool_head="off")
    run.rename(tmp_path / "baseline_run")
    current = _resume_checkout(CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
                               stable_tool_head="off")
    assert current == baseline
