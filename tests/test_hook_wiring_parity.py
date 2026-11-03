"""Raw no-hooks parity against an untouched checkout of the base release.

Set SF_TEST_PARITY_BASELINE to the release this branch is based on. The name stays outside the
STATEFRAME_ namespace: the CLI reports every unknown STATEFRAME_ variable, which would add a line to
the output of every CLI test in the same session. Both sides explicitly disable the stable tool head:
the base release may default it on (0.3.13 and later), and the parity is about hooks, not tool lists.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from acceptance._support import assert_tree_imports, tree_python

import kernel_measurement as measurement
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve


CURRENT = Path(__file__).resolve().parents[1]
_baseline_path = os.environ.get("SF_TEST_PARITY_BASELINE")
BASELINE = Path(_baseline_path).expanduser().resolve() if _baseline_path else None
RUNNER_SPEC_FIELDS = {"check_cmd", "check_source", "protect_tests",
                      "protected_test_globs", "runner_kind"}
pytestmark = [
    pytest.mark.host_sandbox,
    pytest.mark.skipif(
        BASELINE is None,
        reason="set SF_TEST_PARITY_BASELINE to an untouched base-release checkout",
    ),
]


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


def test_no_hooks_raw_requests_and_journal_match_release(tmp_path):
    assert (BASELINE / "src/stateframe/runner.py").is_file()
    repo = tmp_path / "repo"
    source = CURRENT / "examples/discount_bug"
    import shutil

    shutil.copytree(source, repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(
        BASELINE, repo, run, launcher, wrapper, tmp_path / "baseline_modules.json", stable_tool_head="off",
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


def test_no_hooks_regular_stateframe_file_matches_release(tmp_path):
    repo = tmp_path / "repo"
    import shutil

    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    (repo / ".stateframe").write_text("ordinary repository data", encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(BASELINE, repo, run, launcher, wrapper, tmp_path / "baseline.json",
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
        os.kill(int(children[0]), signal.SIGKILL)
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
            process.kill()
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()


def test_no_hooks_resume_rebuilt_requests_match_release(tmp_path):
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _resume_checkout(BASELINE, repo, run, launcher, wrapper, tmp_path / "baseline.json",
                                stable_tool_head="off")
    run.rename(tmp_path / "baseline_run")
    current = _resume_checkout(CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
                               stable_tool_head="off")
    assert current == baseline
