"""L0 wire parity checks against the unmodified base checkout."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time

import pytest

from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from mutate_leftovers import mutation_root, sweep_and_assert


CURRENT = Path(__file__).resolve().parents[1]
_baseline = os.environ.get("SF_TEST_PARITY_BASELINE")
BASELINE = Path(_baseline).expanduser().resolve() if _baseline else None
EXAMPLES = {
    "discount_bug": "Fix the pricing code so the tests pass.",
    "tiny_site": "Create the product page and run its tests.",
    "brief_to_report": "Turn the brief into the requested report and verify it.",
}


def _write_trace_launchers(launcher: Path, wrapper: Path) -> None:
    launcher.write_text(
        "import secrets, subprocess, sys\n"
        "_sequence = 0\n"
        "def fixed_hex(size=32):\n"
        "    global _sequence\n"
        "    _sequence += 1\n"
        "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
        "secrets.token_hex = fixed_hex\n"
        "from stateframe import cli\n"
        "wrapper, trace, child_pid, *args = sys.argv[1:]\n"
        "original_popen = subprocess.Popen\n"
        "def traced_popen(command, *positional, **keywords):\n"
        "    if len(command) == 5 and list(command[1:3]) == ['-B', '-c'] and command[4] == 'stateframe._child':\n"
        "        import ast, os, re\n"
        "        child_path, run_path = ast.literal_eval(re.search(r'sys\\.argv=(\\[.*?\\]);', command[3]).group(1))\n"
        "        command = [os.path.join(keywords.get('cwd') or '.', command[0]), '-B', child_path, run_path]\n"
        "        keywords['cwd'] = run_path\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        process = original_popen([command[0], command[1], wrapper, command[2], command[3], trace], *positional, **keywords)\n"
        "        open(child_pid, 'w').write(str(process.pid))\n"
        "        return process\n"
        "    return original_popen(command, *positional, **keywords)\n"
        "subprocess.Popen = traced_popen\n"
        "raise SystemExit(cli.main(args))\n",
        encoding="utf-8",
    )
    wrapper.write_text(
        "import json, os, runpy, sys\n"
        "child, run, trace = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.argv = [child, run]\n"
        "try: runpy.run_path(child, run_name='__main__')\n"
        "finally: open(trace, 'w').write(json.dumps(sorted(sys.modules)))\n",
        encoding="utf-8",
    )


def _remove_generated_run(run: Path) -> None:
    # Receipt IDs include the absolute run path, so both captures use this fixture path.
    for directory, children, _files in os.walk(run):
        Path(directory).chmod(0o700)
        for child in children:
            path = Path(directory) / child
            if not path.is_symlink():
                path.chmod(0o700)
    shutil.rmtree(run)


def _run(checkout: Path, repo: Path, run: Path, example: str, stable_head: str,
         hooks: Path, launcher: Path, wrapper: Path, trace: Path,
         mutation_value: str | None = None):
    script_path = CURRENT / "examples" / f"{example}.script.json"
    script = Script(json.loads(script_path.read_text(encoding="utf-8")))
    server = serve(script)
    environment = {key: value for key, value in os.environ.items()
                   if key != "STATEFRAME_MUTATIONS"}
    environment.update({"PYTHONPATH": str(checkout / "src"),
                        "STATEFRAME_STABLE_TOOL_HEAD": stable_head})
    if mutation_value is not None:
        environment["STATEFRAME_MUTATIONS"] = mutation_value
    try:
        result = subprocess.run([
            sys.executable, str(launcher), str(wrapper), str(trace), str(trace) + ".child", "run",
            "--repo", str(repo), "--task", EXAMPLES[example],
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
            "--max-turns", "32", "--quiet", "--hooks", str(hooks),
        ], env=environment, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        requests = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert requests
        modules = set(json.loads(trace.read_text(encoding="utf-8")))
        return requests, modules
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(BASELINE is None,
                    reason="set SF_TEST_PARITY_BASELINE to the untouched base checkout")
@pytest.mark.parametrize("example", tuple(EXAMPLES))
@pytest.mark.parametrize("stable_head", ("off", "on"))
@pytest.mark.parametrize("mutation_mode", ("unset", "off"))
def test_mutation_l0_request_bytes_and_stable_tool_head_parity(
        mutation_root, tmp_path, example, stable_head, mutation_mode):
    assert BASELINE is not None and (BASELINE / "src/stateframe/runner.py").is_file()
    repo = tmp_path / "parity-repo"
    run = tmp_path / "parity-run"
    shutil.copytree(CURRENT / "examples" / example, repo)
    hook = tmp_path / "observer.py"
    hook.write_text("pass\n", encoding="utf-8")
    hooks = tmp_path / "hooks.json"
    hooks.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "observe", "event": "post_tool", "argv": [sys.executable, "observer.py"],
        "files": ["observer.py"],
    }]}), encoding="utf-8")
    launcher, wrapper = tmp_path / "launcher.py", tmp_path / "wrapper.py"
    _write_trace_launchers(launcher, wrapper)

    base = _run(BASELINE, repo, run, example, stable_head, hooks,
                launcher, wrapper, tmp_path / "base-modules.json")
    shutil.rmtree(repo)
    _remove_generated_run(run)
    shutil.copytree(CURRENT / "examples" / example, repo)
    current = _run(CURRENT, repo, run, example, stable_head, hooks,
                   launcher, wrapper, tmp_path / "current-modules.json",
                   mutation_value="off" if mutation_mode == "off" else None)
    assert current[0] == base[0]
    assert not any(name.startswith("kernel_mutate") or name.endswith(".mutate_host")
                   or name == "mutate_host" for name in current[1])
    sweep_and_assert(mutation_root)


def _resume_run(checkout: Path, repo: Path, run: Path, stable_head: str,
                hooks: Path, launcher: Path, wrapper: Path, trace: Path,
                mutation_value: str | None = None):
    class InterruptedScript(Script):
        def __init__(self):
            super().__init__(json.loads((CURRENT / "examples/discount_bug.script.json")
                                        .read_text(encoding="utf-8")))
            self.waiting = threading.Event()
            self.release = threading.Event()

        def next_call(self, offered):
            if self.position == 2 and not self.release.is_set():
                self.waiting.set()
                if not self.release.wait(600):
                    raise TimeoutError("resume parity barrier was not released")
                raise ScriptStepUnavailable("request interrupted before a response")
            return super().next_call(offered)

    script = InterruptedScript()
    server = serve(script)
    environment = {key: value for key, value in os.environ.items()
                   if key != "STATEFRAME_MUTATIONS"}
    environment.update({"PYTHONPATH": str(checkout / "src"),
                        "STATEFRAME_STABLE_TOOL_HEAD": stable_head})
    if mutation_value is not None:
        environment["STATEFRAME_MUTATIONS"] = mutation_value
    child_pid = Path(str(trace) + ".child")
    command = [sys.executable, str(launcher), str(wrapper), str(trace), str(child_pid), "run",
               "--repo", str(repo), "--task", EXAMPLES["discount_bug"],
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "16",
               "--quiet", "--hooks", str(hooks)]
    launched_at = time.time()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert script.waiting.wait(600), "first attempt did not reach interrupted request"
        original = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
                    if path.with_name(path.name.replace(".request.", ".response.")).exists()]
        assert len(original) == 2
        pid = int(child_pid.read_text(encoding="ascii"))
        identity = record_identity(pid, launched_after=launched_at,
                                   command_fragment="_child")
        signal_run_verified(run, pid, identity, signal.SIGKILL,
                            deadline=time.monotonic() + 30,
                            what="interrupted parity request")
        script.release.set()
        process.communicate(timeout=30)
        resumed = subprocess.run(
            [sys.executable, str(launcher), str(wrapper), str(trace), str(child_pid),
             "resume", str(run), "--endpoint",
             f"http://127.0.0.1:{server.server_port}/v1"],
            env=environment, capture_output=True, text=True, timeout=600)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        rebuilt = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        return original, rebuilt, set(json.loads(trace.read_text(encoding="utf-8")))
    finally:
        script.release.set()
        if process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="parity cleanup", require_checkpoint_saved=False)
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
@pytest.mark.skipif(BASELINE is None,
                    reason="set SF_TEST_PARITY_BASELINE to the untouched base checkout")
@pytest.mark.parametrize("stable_head", ("off", "on"))
@pytest.mark.parametrize("mutation_mode", ("unset", "off"))
def test_mutation_l0_resume_selection_parity(mutation_root, tmp_path, stable_head,
                                              mutation_mode):
    assert BASELINE is not None
    launcher, wrapper = tmp_path / "launcher.py", tmp_path / "wrapper.py"
    _write_trace_launchers(launcher, wrapper)
    hook = tmp_path / "observer.py"
    hook.write_text("pass\n", encoding="utf-8")
    hooks = tmp_path / "hooks.json"
    hooks.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [{
        "name": "observe", "event": "post_tool", "argv": [sys.executable, "observer.py"],
        "files": ["observer.py"],
    }]}), encoding="utf-8")
    repo = tmp_path / "parity-repo"
    run = tmp_path / "parity-run"
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    baseline = _resume_run(BASELINE, repo, run, stable_head,
                           hooks, launcher, wrapper, tmp_path / "baseline-modules.json")
    shutil.rmtree(repo)
    _remove_generated_run(run)
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    current = _resume_run(CURRENT, repo, run, stable_head,
                          hooks, launcher, wrapper, tmp_path / "current-modules.json",
                          mutation_value="off" if mutation_mode == "off" else None)
    assert current[:2] == baseline[:2]
    assert not any(name.startswith("kernel_mutate") or name.endswith(".mutate_host")
                   or name == "mutate_host" for name in current[2])
    sweep_and_assert(mutation_root)
