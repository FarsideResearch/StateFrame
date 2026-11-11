"""Inactive powerups leave provider bytes and spec shape unchanged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _timeouts import scaled

from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def test_feature_off_raw_request_parity(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "test_sample.py").write_text(
        "from sample import VALUE\ndef test_value():\n    assert VALUE == 1\n"
    )
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import secrets,sys\nfrom stateframe import cli\n"
        "original=secrets.token_hex\nused=False\n"
        "def fixed(size=None):\n global used\n"
        " if size==16 and not used:\n  used=True\n  return 'a'*32\n"
        " return original(size)\n"
        "secrets.token_hex=fixed\nraise SystemExit(cli.main(sys.argv[1:]))\n"
    )
    document = {
        "model": "scripted",
        "steps": [
            {"tool": "run_tests", "arguments": {"target": "test_sample.py"}},
            {"tool": "done", "arguments": {"summary": "Checks passed."}},
        ],
        "tokenize": {"mode": "ok"},
    }
    script = Script(document)
    server = serve(script)
    run = tmp_path / "run"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "STATEFRAME_HOME": str(tmp_path / "home"),
    }

    def execute(extra):
        result = subprocess.run(
            [
                sys.executable,
                str(launcher),
                "run",
                "--repo",
                str(repo),
                "--task",
                "Inspect VALUE.",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "scripted",
                "--run-dir",
                str(run),
                "--test-python",
                sys.executable,
                "--no-run-env",
                "--shell",
                "off",
                "--max-turns",
                "4",
                "--quiet",
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=scaled(600),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        spec = json.loads((run / "spec.json").read_text())
        wire = [
            path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
        ]
        assert wire
        return spec, wire

    try:
        ordinary = execute([])
        run.rename(tmp_path / "ordinary_run")
        script.__init__(document)
        disabled = execute(["--no-powerups"])
        assert "powerups" not in ordinary[0] and "powerups" not in disabled[0]
        assert set(ordinary[0]) == set(disabled[0])
        assert ordinary[1] == disabled[1]
    finally:
        server.shutdown()
        server.server_close()
