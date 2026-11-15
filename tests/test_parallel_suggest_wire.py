"""A real parent send carries the selected outlines and host recommendation."""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored

from stateframe.scripted_endpoint import Script, serve


@pytest.mark.host_sandbox
def test_parallel_suggest_on_first_wire(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "a.py").write_text("SECRET_A = 123\ndef a(): return SECRET_A\n")
    (repo / "b.py").write_text("SECRET_B = 456\ndef b(): return SECRET_B\n")
    (repo / "tests/test_a.py").write_text(
        "from a import a\ndef test_a(): assert a() == 123\n")
    (repo / "tests/test_b.py").write_text(
        "from b import b\ndef test_b(): assert b() == 456\n")
    home = tmp_path / "home"
    home.mkdir()
    run = tmp_path / "run"
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"},
                     "steps": [{"tool": "done", "required": True,
                                "arguments": {"summary": "Checked both modules."}}]})
    server = serve(script)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONHASHSEED": "0", "HOME": str(home),
        "STATEFRAME_HOME": isolated_stateframe_home(),
        # The suggestion is applied by the legacy frame allocation; the working-set
        # allocator (shipped on) does not consume it yet.
        "STATEFRAME_WORKING_SET_ALLOCATOR": "off",
        "STATEFRAME_PARALLEL_SUGGEST": "on",
        "STATEFRAME_PARALLEL_SUGGEST_BACKEND": "hosted",
    })
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
               "--task", "Check independent a.py and b.py against their own tests.",
               "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
               "--model", "scripted", "--run-dir", str(run),
               "--test-python", sys.executable,
               "--test-cmd", shlex.join((sys.executable, "-m", "pytest", "-q",
                                           "tests/test_a.py", "tests/test_b.py")),
               "--parallel", "2", "--max-turns", "1", "--quiet"]
    try:
        result = run_monitored(command, cwd=repo, env=environment, run_dir=run,
                               expected_seconds=300, step="parallel suggestion wire",
                               check=False)
        wires = sorted((run / "wire").glob("*.request.json"))
        assert wires, result.stdout + result.stderr
        request = json.loads(wires[0].read_bytes())
        contents = "\n".join(str(message.get("content", ""))
                             for message in request["messages"])
        assert "Independent pairs: a.py<->tests/test_a.py; b.py<->tests/test_b.py" in contents
        assert "[Parallel outline: a.py]" in contents
        assert "[Parallel outline: tests/test_b.py]" in contents
        assert "SECRET_A = 123" not in contents
        assert "SECRET_B = 456" not in contents
        assert "Current source bytes attached to this request: a.py" not in contents
        assert "- a.py:1-2 revision=unknown" not in contents
        assert contents.index("Parallel selection:") < contents.index("Independent pairs:")
        log = (run / "engine.log").read_text()
        assert "parallel suggestion hosted:axis=" in log
        assert "recommendation=parallel" in log
    finally:
        server.shutdown()
        server.server_close()
