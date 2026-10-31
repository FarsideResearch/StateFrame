"""A run without MCP servers sends the same request bytes as release 0.2.1.

The golden file was captured from the 0.2.1 tree (97cd52d) with the offline
discount_bug script. Edit receipts hash the run path, so captures use one fixed
run root, serialized by a lock; only the test interpreter path is normalized.
"""
import fcntl
import os
import shutil
import sys
from pathlib import Path

import pytest
from tools.compare_no_mcp_requests import capture, write_launcher

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "fixtures/no_mcp_0_2_1_requests.bin"
RUN_ROOT = Path("/tmp") / f"sf-no-mcp-golden-{os.getuid()}"


@pytest.mark.host_sandbox
def test_no_mcp_requests_match_release_0_2_1():
    with open(f"{RUN_ROOT}.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if RUN_ROOT.exists():
            shutil.rmtree(RUN_ROOT)
        RUN_ROOT.mkdir()
        try:
            root = RUN_ROOT.resolve()
            repo = root / "repo"
            shutil.copytree(ROOT / "examples/discount_bug", repo)
            launcher = root / "fixed_seed_run.py"
            write_launcher(launcher)
            body = capture(ROOT, sys.executable, launcher, ROOT / "examples/discount_bug.script.json",
                           repo, root / "run")
        finally:
            shutil.rmtree(RUN_ROOT, ignore_errors=True)
    body = body.replace(sys.executable.encode(), b"<python>")
    golden = GOLDEN.read_bytes()
    assert len(body) == len(golden)
    assert body == golden
