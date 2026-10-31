"""Compare no-MCP request bytes with the pinned 0.2.1 source tree."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tools.compare_no_mcp_requests import capture, write_launcher

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "97cd52d6c2225a823fdfeb632aad7bfd738f4d91"
BASELINE_TREE = "d8fda320fac7b89387cbd5e5a24d04970f81d989"


def _materialize_baseline(destination: Path) -> str:
    if shutil.which("git") is None:
        pytest.skip("git is unavailable, so the pinned 0.2.1 tree cannot be materialized")
    present = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e",
                              f"{BASELINE_COMMIT}^{{commit}}"], capture_output=True, check=False)
    if present.returncode != 0:
        pytest.skip("pinned 0.2.1 commit 97cd52d is unavailable in this repository's git objects")
    tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{BASELINE_COMMIT}^{{tree}}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert tree == BASELINE_TREE, f"pinned 0.2.1 commit has unexpected tree {tree}"
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar",
                              BASELINE_COMMIT], capture_output=True, check=True).stdout
    destination.mkdir()
    subprocess.run(["tar", "-xf", "-", "-C", str(destination)], input=archive, check=True)
    return tree


@pytest.mark.host_sandbox
def test_no_mcp_requests_match_release_0_2_1(tmp_path):
    baseline = tmp_path / "baseline"
    tree = _materialize_baseline(baseline)
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "fixed_seed_run.py"
    write_launcher(launcher)
    run_dir = tmp_path / "run"
    script = ROOT / "examples/discount_bug.script.json"
    previous = capture(baseline, sys.executable, launcher, script, repo, run_dir)
    run_dir.rename(tmp_path / "baseline-run")
    current = capture(ROOT, sys.executable, launcher, script, repo, run_dir)
    previous_sha = hashlib.sha256(previous).hexdigest()
    current_sha = hashlib.sha256(current).hexdigest()
    assert previous == current, (f"0.2.1 commit {BASELINE_COMMIT} tree {tree}: "
                                f"sha256={previous_sha}; candidate sha256={current_sha}")
