"""Evicting an environment seed does not change model request bytes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _timeouts import scaled
from stateframe.request_records import list_request_records, read_request_record
from stateframe.scripted_endpoint import Script, serve
from tools.compare_no_mcp_requests import write_launcher


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def _capture(repo: Path, run: Path, launcher: Path, script_path: Path,
             python: Path, home: Path, cap: str) -> tuple[int, bytes]:
    script = Script(json.loads(script_path.read_text()))
    server = serve(script)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"),
                   str(ROOT / "src/stateframe/engine"))),
                   "HOME": str(home), "STATEFRAME_HOME": str(home / ".stateframe"),
                   "STATEFRAME_ENV_SEED_MAX_GB": cap}
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), "run", "--repo", str(repo),
             "--task", "Complete the example task and run its checks.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--test-python", str(python),
             "--test-cmd", "python -m unittest discover -s tests -q",
             "--run-env", "--max-turns", "32", "--quiet"],
            # One capture took 186 s solo and timed out at 300 s under concurrent Linux gate load
            # (11-04); 900 s keeps headroom without letting a hung capture run unbounded.
            env=environment, capture_output=True, text=True, timeout=scaled(900))
        # Parity concerns request bytes, not the example's check outcome under this interpreter.
        assert result.returncode in (0, 1), result.stdout + result.stderr
        script.assert_consumed()
        assert (run / "env-seed.json").is_file(), result.stdout + result.stderr
        names = list_request_records(run)
        assert names
        return result.returncode, b"\n".join(read_request_record(run, name) for name in names)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("example", ("discount_bug", "tiny_site", "brief_to_report"))
def test_env_seed_cap_eviction_preserves_request_bytes(tmp_path, clean_test_environment, example):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    python = clean_test_environment(tmp_path / "test-python")
    launcher = tmp_path / "fixed-seed-run.py"
    write_launcher(launcher)
    home = tmp_path / "home"
    store = home / ".stateframe" / "env-seeds"
    store.mkdir(parents=True)
    script = ROOT / "examples" / f"{example}.script.json"
    run = tmp_path / "run"
    baseline = _capture(repo, run, launcher, script, python, home, "0")
    run.rename(tmp_path / "baseline-run")

    # This unrelated old seed makes the second clone cross the cap, while the active seed stays.
    old_seed = store / ("f" * 64)
    old_seed.mkdir()
    (old_seed / "payload").write_bytes(b"old seed")
    capped = _capture(repo, run, launcher, script, python, home, "0.000000001")

    assert not old_seed.exists(), "the capped run did not exercise eviction"
    assert capped == baseline
