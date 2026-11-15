"""Ambient subframe settings cannot change an ordinary run or its wire bytes."""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from _stateframe_home import isolated_stateframe_home

from stateframe import runner
from stateframe.scripted_endpoint import Script, make_handler, serve

ROOT = Path(__file__).resolve().parents[1]
PARALLEL_VALUES = {
    name: choices[0] if choices else
    "25" if name == "STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT" else "missing.json"
    for name, choices in runner.OPT_IN_SWITCHES.items()
    if name.startswith("STATEFRAME_PARALLEL_")
}
PARALLEL_VALUES.update({
    "STATEFRAME_PARALLEL_USAGE_FEEDBACK": "off",
    "STATEFRAME_PARALLEL_FANOUT_CONTINUATION": "on",
})


def _ambient_values(kind: str) -> dict[str, str]:
    if kind == "valid":
        return dict(PARALLEL_VALUES)
    return {name: "x" if name != "STATEFRAME_PARALLEL_FORCE_MANIFEST" else "x" * 4097
            for name in PARALLEL_VALUES}


@pytest.mark.parametrize("name", PARALLEL_VALUES)
@pytest.mark.parametrize("kind", ("valid", "malformed"))
@pytest.mark.parametrize("boundary", ("resolver", "provider"))
def test_parallel_off_ignores_each_ambient_switch(monkeypatch, tmp_path, name, kind, boundary):
    for switch in tuple(os.environ):
        if switch.startswith("STATEFRAME_"):
            monkeypatch.delenv(switch)
    options = runner.RunOptions(tmp_path, "inspect", None, parallel_mode="off")

    def resolve():
        if boundary == "resolver":
            return runner.resolve_opt_in_switches(parallel_enabled=False)
        return runner._provider_environment(options)

    baseline = resolve()
    monkeypatch.setenv(name, _ambient_values(kind)[name])
    assert resolve() == baseline


@pytest.mark.parametrize("name", PARALLEL_VALUES)
def test_parallel_on_still_rejects_malformed_ambient_switch(monkeypatch, name):
    for switch in tuple(os.environ):
        if switch.startswith("STATEFRAME_"):
            monkeypatch.delenv(switch)
    monkeypatch.setenv(name, _ambient_values("malformed")[name])
    with pytest.raises(runner.RunError, match=name):
        runner.resolve_opt_in_switches(parallel_enabled=True)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("parallel_args", ((), ("--parallel", "off")), ids=("omitted", "off"))
@pytest.mark.parametrize("kind", ("valid", "malformed"))
def test_t1_parallel_off_golden_wire_ignores_ambient(tmp_path, parallel_args, kind, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value.py").write_text("VALUE = 1\n")
    # A whole-command check has deterministic output, including on failure.
    (repo / "check.py").write_text(
        "from value import VALUE\n"
        "print('PASS' if VALUE == 2 else 'FAIL')\n"
        "raise SystemExit(0 if VALUE == 2 else 1)\n"
    )
    launcher = tmp_path / "fixed-run.py"
    launcher.write_text(
        "import os, runpy, secrets, subprocess, sys, time\n"
        "time.time = lambda: 1700000000.0\n"
        "if sys.argv[1] == '--child':\n"
        "    child = sys.argv.pop(2)\n"
        "    sys.argv = [child, *sys.argv[2:]]\n"
        "    sys.path.insert(0, os.path.dirname(child))\n"
        "    runpy.run_path(child, run_name='__main__')\n"
        "else:\n"
        "    original_hex = secrets.token_hex\n"
        "    secrets.token_hex = lambda n=32: 'a' * 32 if n == 16 else original_hex(n)\n"
        "    original_popen = subprocess.Popen\n"
        "    def popen(command, *args, **kwargs):\n"
        "        if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "            command = [*command[:2], __file__, '--child', *command[2:]]\n"
        "        return original_popen(command, *args, **kwargs)\n"
        "    subprocess.Popen = popen\n"
        "    runpy.run_module('stateframe', run_name='__main__')\n"
    )
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("STATEFRAME_")}
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home/.stateframe"))
    environment.update({
        "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": "0",
        "HOME": str(tmp_path / "home"), "STATEFRAME_HOME": isolated_stateframe_home(),
    })
    Path(environment["HOME"]).mkdir()
    run = tmp_path / "run"
    document = {"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "run_tests", "required": True, "arguments": {}},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "value.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Fixed the value."}},
    ]}
    script = Script(document)
    server = serve(script)
    command = [sys.executable, str(launcher), "run", "--repo", str(repo),
               "--task", "Set VALUE to 2 in value.py and check it with check.py.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--run-dir", str(run), "--test-python", sys.executable,
               "--test-cmd", shlex.join((sys.executable, "check.py")),
               "--max-turns", "3", "--shell", "off",
               "--quiet", *parallel_args]

    def capture(extra):
        completed = run_monitored(
            command, cwd=repo, env={**environment, **extra}, run_dir=run,
            expected_seconds=float(os.environ.get("SF_TEST_ENGINE_TIMEOUT", "300")),
            step="T1 parallel-off golden wire", check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        script.assert_consumed()
        summary = json.loads((run / "summary.json").read_text())
        assert summary["tests_before"]["passed"] is False
        assert summary["tests_after"]["passed"] is True
        wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert len(wires) == 3
        spec = json.loads((run / "spec.json").read_text())
        assert not any(name.startswith("STATEFRAME_PARALLEL_") for name in spec["environment"])
        assert not (run / "parallel").exists()
        assert all(tool["function"]["name"] != "parallel"
                   for wire in wires for tool in json.loads(wire)["tools"])
        assert any(message["role"] == "tool" and "check exit 1" in message["content"]
                   and "\nFAIL" in message["content"]
                   for message in json.loads(wires[1])["messages"]), (
                       "second send did not observe the failed check")
        assert any(message["role"] == "tool" and message["content"] == "str_replace applied."
                   for message in json.loads(wires[2])["messages"]), (
                       "third send did not observe the applied edit")
        return wires

    try:
        clean = capture({})
        run.rename(tmp_path / "clean-run")
        # Reuse the same run path and endpoint so exact bytes need no normalization.
        script = Script(document)
        server.RequestHandlerClass = make_handler(script)
        hostile = capture(_ambient_values(kind))
        for send, (expected, actual) in enumerate(zip(clean, hostile), 1):
            assert actual == expected, f"T1 raw request bytes differ at send {send}"
    finally:
        server.shutdown()
        server.server_close()
