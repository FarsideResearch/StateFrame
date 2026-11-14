"""Subframe CLI wrappers retain shared monitoring and captured diagnostics."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import e2e_support
import pytest
from engine_run import EngineProcess, run_monitored, start_monitored

from stateframe import runner
from stateframe.cli import build_parser


def test_wait_for_parent_wire_observes_replayed_request_without_provider_arrival(tmp_path):
    class Running:
        def poll(self):
            return None

    request = tmp_path / "wire/0002.request.json"
    request.parent.mkdir()
    timer = threading.Timer(0.05, lambda: request.write_text("{}"))
    timer.start()
    try:
        assert e2e_support.wait_for_parent_wire(tmp_path, 2, Running(), timeout=1) == request
    finally:
        timer.join()


def test_crash_hook_runs_when_sitecustomize_is_already_loaded(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)],
                   capture_output=True, text=True, timeout=90, check=True)
    python = venv / "bin/python"
    site, = (venv / "lib").glob("python*/site-packages")
    # A distribution's stdlib sitecustomize takes precedence over a venv's.
    # Simulate an existing module even on hosts without that customization.
    (site / "00_existing_sitecustomize.pth").write_text(
        "import sys, types; sys.modules.setdefault('sitecustomize', "
        "types.ModuleType('sitecustomize'))\n")
    hook = e2e_support.install_crash_hook(python)
    assert e2e_support.install_crash_hook(python) == hook
    marker = tmp_path / "marker"
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(e2e_support.ROOT / 'src/stateframe/engine')!r})\n"
        # Match _child: the spec environment arrives after interpreter startup.
        "os.environ['SF_PARALLEL_E2E_CRASH'] = 'after_input_refusal_intent'\n"
        f"os.environ['SF_PARALLEL_E2E_CRASH_MARKER'] = {str(marker)!r}\n"
        "from kernel_tool_tape import ToolTape\n"
        f"tape = ToolTape(Path({str(tmp_path)!r}))\n"
        "tape.intent(call_id='probe', tool='parallel', "
        "args={'schema': 'stateframe-parallel-input-refusal/1'}, replay_class='local', "
        "capability_digest='probe', definition_digest=None, env_identity='probe', "
        "pre_manifest_digest='probe')\n"
    )
    result = subprocess.run([str(python), "-I", "-c", code], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 86, result.stdout + result.stderr
    assert marker.read_text() == "after_input_refusal_intent"
    pending = e2e_support.read_json(tmp_path / "tool-tape.jsonl")
    assert pending["kind"] == "intent"

    materialized_marker = tmp_path / "materialized-marker"
    parent_file = tmp_path / "run/worker/continuation-workspaces/active/repo/pkg/a.py"
    unrelated_file = tmp_path / "unrelated/pkg/a.py"
    materialize_code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(e2e_support.ROOT / 'src/stateframe/engine')!r})\n"
        "os.environ['SF_PARALLEL_E2E_CRASH'] = 'between_materialized_files'\n"
        f"os.environ['SF_PARALLEL_E2E_CRASH_MARKER'] = {str(materialized_marker)!r}\n"
        "import kernel_parallel_txn\n"
        f"for target in (Path({str(parent_file)!r}), Path({str(unrelated_file)!r})):\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text('before')\n"
        "    (target.parent / '.parallel-probe').write_text('after')\n"
        f"os.replace(Path({str(unrelated_file.parent)!r}) / '.parallel-probe', "
        f"Path({str(unrelated_file)!r}))\n"
        f"os.replace(Path({str(parent_file.parent)!r}) / '.parallel-probe', "
        f"Path({str(parent_file)!r}))\n"
    )
    materialized = subprocess.run([str(python), "-I", "-c", materialize_code],
                                  capture_output=True, text=True, timeout=30, check=False)
    assert materialized.returncode == 86, materialized.stdout + materialized.stderr
    assert unrelated_file.read_text() == "after"
    assert parent_file.read_text() == "after"
    assert materialized_marker.read_text() == "between_materialized_files"
    recovered = subprocess.run([str(python), "-I", "-c", materialize_code],
                               capture_output=True, text=True, timeout=30, check=False)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert materialized_marker.read_text() == "between_materialized_files"


def test_crash_hook_after_ack_forwards_outcome_recording_arguments(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)],
                   capture_output=True, text=True, timeout=90, check=True)
    python = venv / "bin/python"
    site, = (venv / "lib").glob("python*/site-packages")
    (site / "00_existing_sitecustomize.pth").write_text(
        "import sys, types; sys.modules.setdefault('sitecustomize', "
        "types.ModuleType('sitecustomize'))\n")
    e2e_support.install_crash_hook(python)
    marker = tmp_path / "ack-marker"
    call_root = tmp_path / "call"
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(e2e_support.ROOT / 'src/stateframe/engine')!r})\n"
        "os.environ['SF_PARALLEL_E2E_CRASH'] = 'after_ack'\n"
        f"os.environ['SF_PARALLEL_E2E_CRASH_MARKER'] = {str(marker)!r}\n"
        "import kernel_parallel\n"
        "from kernel_parallel_records import UnitOutcome, UnitUsage\n"
        "class Budget:\n"
        "    def terminal(self, call_id, unit_id, *, parent_steps_spent):\n"
        "        assert (call_id, unit_id, parent_steps_spent) == ('call', 'pair', 0)\n"
        "outcome = UnitOutcome('pair', 'COMMITTED_UNVERIFIED', None, None, 1, (), None, "
        "'NOT_RUN', UnitUsage(0, 0, 0, 0, None), 0.0, 0.0)\n"
        f"kernel_parallel._record_outcome(Path({str(call_root)!r}), outcome, Budget(), 0)\n"
    )
    result = subprocess.run([str(python), "-I", "-c", code], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 86, result.stdout + result.stderr
    assert marker.read_text() == "after_ack"
    recorded, = (call_root / "outcomes.jsonl").read_text().splitlines()
    assert json.loads(recorded)["txn_id"] == 1


def test_crash_hook_arguments_use_test_env_instead_of_filtered_ambient_env(tmp_path, monkeypatch):
    marker = tmp_path / "crash-marker"
    command = e2e_support.cli_command(
        Path(sys.executable), "http://127.0.0.1:8080/v1", tmp_path / "repo",
        tmp_path / "run", "test task",
        arguments=e2e_support.crash_hook_arguments("after_decided", marker),
    )
    parsed = build_parser().parse_args(command[4:])
    expected = {
        "SF_PARALLEL_E2E_CRASH": "after_decided",
        "SF_PARALLEL_E2E_CRASH_MARKER": str(marker),
    }
    assert runner.validate_test_env(dict(parsed.test_env)) == expected
    monkeypatch.setenv("SF_PARALLEL_E2E_CRASH", "after_decided")
    monkeypatch.setenv("SF_PARALLEL_E2E_CRASH_MARKER", str(marker))
    child_env = runner._child_env(runner.RunOptions(repo=tmp_path, task="test task", endpoint=None))
    assert expected.keys().isdisjoint(child_env)


@pytest.mark.parametrize("owner", [
    {"schema": "stateframe-framefit5-owner/1"},
    {"schema": "stateframe-framefit5-owner/1", "admitted_actions": ["read", "done"]},
])
def test_subframe_surface_accepts_the_shipped_owner_menu_shapes(tmp_path, owner):
    request = tmp_path / "request.json"
    frame = {"schema": "stateframe-f4/1", "state": {"framefit5": {
        "mode": "FRAME_ONLY", "owner_protocol": owner}}}
    request.write_text(json.dumps({
        "messages": [{"content": "[StateFrame]\n" + json.dumps(frame)}],
        "tools": [{"function": {"name": name}} for name in ("read", "done")],
    }))

    assert e2e_support.assert_subframe_surface(request)["schema"] == "stateframe-f4/1"


def test_subframe_surface_rejects_partial_owner_menu(tmp_path):
    request = tmp_path / "request.json"
    frame = {"schema": "stateframe-f4/1", "state": {"framefit5": {
        "mode": "FRAME_ONLY", "owner_protocol": {
            "schema": "stateframe-framefit5-owner/1", "route": "ACTION"}}}}
    request.write_text(json.dumps({
        "messages": [{"content": "[StateFrame]\n" + json.dumps(frame)}],
        "tools": [{"function": {"name": "read"}}],
    }))

    with pytest.raises(AssertionError, match="owner menu is incomplete"):
        e2e_support.assert_subframe_surface(request)


@pytest.mark.parametrize("mode", ["run", "start", "resume", "resume_capture"])
def test_cli_wrappers_use_shared_monitor_and_preserve_process_controls(tmp_path, monkeypatch, mode):
    run_dir = tmp_path / "run"
    repo = tmp_path / "repo"
    python = Path(sys.executable)
    endpoint = "http://127.0.0.1:8080/v1"
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    monkeypatch.setenv("STATEFRAME_PARALLEL_MODE", "must-not-leak")
    calls = []

    def monitored(command, **kwargs):
        calls.append((command, kwargs))
        # Exercise the actual watcher without starting a model server or engine.
        probe = [sys.executable, "-I", "-c", (
            "import os, sys; print(os.environ['HOME']); "
            "print(os.environ['SF_MONITOR_TEST']); "
            "print('diagnostic', file=sys.stderr); sys.exit(7)")]
        monitor = run_monitored if mode == "run" else start_monitored
        return monitor(probe, **kwargs)

    monkeypatch.setattr(e2e_support, "run_monitored" if mode == "run" else "start_monitored",
                        monitored)
    environment = {"SF_MONITOR_TEST": "forwarded"}
    if mode == "run":
        result = e2e_support.run_cli(python, endpoint, repo, run_dir, "test task",
                                    extra_environment=environment, timeout=123)
        stdout, stderr = result.stdout, result.stderr
        returncode = result.returncode
    else:
        if mode == "start":
            process = e2e_support.start_cli(python, endpoint, repo, run_dir, "test task",
                                           extra_environment=environment)
        else:
            process = e2e_support.start_resume(python, endpoint, run_dir,
                                              extra_environment=environment,
                                              capture_output=mode == "resume_capture")
        assert isinstance(process, EngineProcess)
        with process:
            assert os.getpgid(process.pid) == process.pid
            stdout, stderr = process.communicate(timeout=10)
            returncode = process.poll()

    assert returncode == 7
    assert stdout.splitlines() == [str(tmp_path / "home"), "forwarded"]
    assert stderr == "diagnostic\n"
    (command, options), = calls
    assert options["run_dir"] == run_dir
    assert options["expected_seconds"] == (123 if mode == "run" else 900)
    assert options["cwd"] == e2e_support.ROOT
    assert "PYTHONPATH" not in options["env"]
    assert "STATEFRAME_PARALLEL_MODE" not in options["env"]
    if mode.startswith("resume"):
        assert command == [str(python), "-I", "-m", "stateframe", "resume", str(run_dir),
                           "--endpoint", endpoint, "--model", "scripted", "--keep-checkpoints", "--json"]
    else:
        assert command == e2e_support.cli_command(python, endpoint, repo, run_dir, "test task")
