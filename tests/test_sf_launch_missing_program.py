"""The exec wrapper reports a missing program the way bubblewrap does.

Outer checks name an absent check program ("check command not found: make") by parsing bubblewrap's
"execvp NAME: No such file or directory" line. Since 0.2.4 the Linux sandbox execs commands through
sf_launch.py (relative process limit), so the wrapper must keep that contract or a missing program is
reported as an ordinary failing check.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import stateframe

LAUNCHER = Path(stateframe.__file__).parent / "engine" / "sf_launch.py"


@pytest.mark.parametrize("mode", ["--cgroup-enter", "--nproc-budget"])
def test_missing_program_matches_bwrap_execvp_line(tmp_path, mode):
    if mode == "--nproc-budget" and not sys.platform.startswith("linux"):
        pytest.skip("relative NPROC limit is Linux-only")
    value = str(tmp_path / "no-cgroup") if mode == "--cgroup-enter" else "8"
    result = subprocess.run(
        [sys.executable, "-I", str(LAUNCHER), mode, value, "sf-no-such-check-program", "test"],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0
    assert result.stderr.startswith("bwrap: execvp sf-no-such-check-program: No such file or directory")
    assert "execution launcher failed" not in result.stderr
