"""CLI startup leaves inspection to the command that reads the run records."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from _timeouts import scaled
from engine_run import run_monitored

from stateframe import cli
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pilot", [None, "off", "on"])
def test_run_dispatch_uses_release_imports_when_disabled(tmp_path, pilot):
    source = Path(cli.__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(source)}
    environment.pop("STATEFRAME_SEND1_WAVE4", None)
    if pilot is not None:
        environment["STATEFRAME_SEND1_WAVE4"] = pilot
    result = subprocess.run(
        [sys.executable, "-c", dedent("""\
            import os
            import sys
            from stateframe import cli

            def run(options, *, printer):
                eager = os.environ.get('STATEFRAME_SEND1_WAVE4') != 'on'
                assert ('stateframe.inspect_run' in sys.modules) == eager
                assert options.task == 'Check the example'
                return {'exit_code': 0}

            cli.run = run
            raise SystemExit(cli.main([
                'run', '--repo', '.', '--task', 'Check the example',
                '--endpoint', 'http://127.0.0.1:8080/v1', '--model', 'scripted', '--json',
            ]))
            """)],
        cwd=tmp_path, env=environment,
        capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b'{\n "exit_code": 0\n}\n'


def test_inspect_reads_overview_and_wire_bytes(tmp_path, capsysbinary):
    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    (run / "requests" / "0001.json").write_text("{}", encoding="utf-8")
    (run / "wire").mkdir()
    wire = b'{"messages":[{"role":"user","content":"Check the example"}]}'
    (run / "wire" / "0001.request.json").write_bytes(wire)

    assert cli.main(["inspect", str(run)]) == 0
    captured = capsysbinary.readouterr()
    assert b"preflight records: 1" in captured.out
    assert captured.err == b""

    assert cli.main(["inspect", str(run), "--send", "1", "--bytes"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == wire
    assert captured.err == b""


@pytest.mark.parametrize("flags", [[], ["--send", "1"]])
def test_inspect_reports_missing_run(tmp_path, capsys, flags):
    run = tmp_path / "missing"
    assert cli.main(["inspect", str(run), *flags]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"stateframe: error: {run} has no wire/ directory; is it a stateframe run directory?\n"
    )


def test_skill_commands_manage_and_report_bundles(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    bundle = tmp_path / "example-skill"
    bundle.mkdir()
    content = b"---\nname: example-skill\ndescription: Check the example\n---\nRead app.py.\n"
    (bundle / "SKILL.md").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    assert cli.main(["skill", "install", str(bundle)]) == 0
    assert capsys.readouterr() == (f"installed example-skill ({digest[:12]})\n", "")
    assert cli.main(["skill", "list"]) == 0
    assert capsys.readouterr() == (f"example-skill\tCheck the example\ttools=no\t{digest}\n", "")
    assert cli.main(["skill", "remove", "example-skill"]) == 0
    assert capsys.readouterr() == ("removed example-skill\n", "")
    assert cli.main(["skill", "remove", "example-skill"]) == 2
    assert capsys.readouterr() == ("", "stateframe: error: skill example-skill missing\n")


@pytest.fixture(scope="module")
def startup_wheel(tmp_path_factory):
    directory = tmp_path_factory.mktemp("startup-wheel")
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import runpy, sys\nrunpy.run_path(sys.argv[1])['build_wheel'](sys.argv[2])",
         str(ROOT / "_build" / "sf_backend.py"), str(directory)],
        capture_output=True, text=True, timeout=scaled(60),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = directory.glob("stateframe-*.whl")
    installer = directory / "installer"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "venv", str(installer)],
        capture_output=True, text=True, timeout=scaled(60),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return wheel, installer / "bin" / "python"


@pytest.mark.host_sandbox
@pytest.mark.parametrize("interpreter", ["current", "python3.14"])
def test_installed_startup_sends_through_bash_from_a_relocated_directory(tmp_path, startup_wheel, interpreter):
    python = sys.executable if interpreter == "current" else shutil.which(interpreter)
    bash = shutil.which("bash")
    if python is None or bash is None:
        pytest.skip(f"requires {interpreter} and bash on this host")

    directory = tmp_path / "relocated wheel é"
    directory.mkdir()
    venv = directory / "environment with spaces"
    installed_python = venv / "bin" / "python"
    wheel, installer = startup_wheel
    for command in (
        [python, "-I", "-m", "venv", "--without-pip", str(venv)],
        [str(installer), "-I", "-m", "pip", "--isolated", "--python", str(installed_python),
         "install", "--no-index", "--no-deps", "--no-cache-dir", "--no-compile",
         "--disable-pip-version-check", str(wheel)],
    ):
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=scaled(60))
        assert result.returncode == 0, result.stdout + result.stderr

    repo = directory / "source"
    repo.mkdir()
    (repo / "sample.py").write_text("def sample():\n    return 42\n", encoding="utf-8")
    run = directory / "run with spaces"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "REACT_", "SF_LAB_", "PYTHON"))}
    env.update(STATEFRAME_SEND1_WAVE4="on", STATEFRAME_CHECKPOINT="off",
               STATEFRAME_HOME=str(directory / "home"), PATH=os.defpath)
    script = Script({"tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "arguments": {"path": "sample.py"}, "required": True},
    ]})
    with serve(script) as server:
        command = [
            bash, "--noprofile", "--norc", "-c", 'exec "$@"', "bash", str(installed_python),
            "-I", "-B", "-m", "stateframe", "run", "--repo", str(repo), "--run-dir", str(run),
            "--task", "Inspect sample.py.", "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "scripted", "--test-python", str(Path(python).resolve()),
            "--check-cmd", "true", "--max-turns", "1", "--quiet",
        ]
        try:
            result = run_monitored(command, env=env, cwd=directory, run_dir=run, step="installed startup")
        finally:
            server.shutdown()

    assert (run / "requests/0001.json").is_file(), result.stdout + result.stderr
    assert (run / "parent-start.json").is_file()
    script.assert_consumed()
    assert [json.loads((run / "wire/0001.request.json").read_bytes())] == script.requests
    spec = json.loads((run / "spec.json").read_bytes())
    assert spec["send1_wave4"] is True
    assert Path(spec["engine_dir"]).is_relative_to(venv)
    assert spec["engine_source"]["shipped_hash"]
