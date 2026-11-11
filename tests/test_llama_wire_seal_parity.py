"""Early seal normalization preserves the released llama.cpp request bytes."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _timeouts import scaled

from _journal import measurement_events
from tests.test_final_cache_layout_host import _parity_launchers
from tests.tools.pinned_tree import pinned_tree
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("example,task", [
    ("discount_bug", "Fix the pricing code so the tests pass."),
    ("tiny_site", "Build the small static shop site described in the README."),
    ("brief_to_report", "Write the report the brief asks for."),
])
def test_bundled_llama_wire_matches_30ce267_and_its_seal(
        tmp_path: Path, tmp_path_factory, example: str, task: str) -> None:
    baseline = pinned_tree("30ce267", tmp_path_factory)
    launcher, wrapper = _parity_launchers(tmp_path)
    outputs: list[list[bytes]] = []
    for source, label in ((baseline, "baseline"), (ROOT, "candidate")):
        repo = tmp_path / "repo"
        if not repo.exists():
            shutil.copytree(ROOT / "examples" / example, repo)
        script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
        server = serve(script)
        run = tmp_path / "run"
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith(("STATEFRAME_", "REACT_"))}
        environment.update({
            "PYTHONPATH": str(source / "src"), "STATEFRAME_FINAL_CACHE_LAYOUT": "off",
            "STATEFRAME_PAGE_LOG": "off", "STATEFRAME_HOME": str(tmp_path / "state-home"),
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        })
        try:
            process = subprocess.run(
                [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
                 "--task", task, "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                 "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
                 "--max-turns", "16", "--quiet"],
                env=environment, capture_output=True, text=True, timeout=scaled(600), check=False)
            assert process.returncode == 0, process.stdout + process.stderr
            script.assert_consumed()
            wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
            assert wires
            if label == "candidate":
                ready = {row.get("body_sha256") for row in measurement_events(run)
                         if row.get("kind") == "request_budget_final_wire"
                         and row.get("status") == "READY"}
                assert all("sha256:" + hashlib.sha256(wire).hexdigest() in ready for wire in wires)
            outputs.append(wires)
            run.rename(tmp_path / f"{label}-run")
        finally:
            server.shutdown()
            server.server_close()
    assert outputs[1] == outputs[0]
