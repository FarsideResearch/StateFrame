"""The example hook interpreter must start inside the check sandbox."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from _timeouts import scaled

from kernel_execution_policy import sandboxed_popen


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "powerups"
pytestmark = pytest.mark.host_sandbox


def _run_in_check_sandbox(command: list[str], root: Path) -> tuple[int, bytes, bytes]:
    process, _report = sandboxed_popen(
        command, root, env={"PATH": os.environ.get("PATH", os.defpath)}, generic_check=True
    )
    try:
        stdout, stderr = process.communicate(timeout=scaled(30))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    return process.returncode, stdout, stderr


def test_test_gate_python_starts_in_check_sandbox(tmp_path):
    # Hook loading resolves a bare argv[0] with shutil.which and pins that file.
    hooks = json.loads((EXAMPLES / "test-gate" / "hooks.json").read_text(encoding="utf-8"))
    program_name = hooks["hooks"][0]["argv"][0]
    assert program_name == "python3"
    program = shutil.which(program_name)
    assert program is not None
    python = str(Path(program).resolve(strict=True))

    code, stdout, stderr = _run_in_check_sandbox(
        [python, "-c", "import sys; print(sys.version_info[:2])"], tmp_path
    )
    assert code == 0, (stdout, stderr)
    major, minor = ast.literal_eval(stdout.decode("utf-8").strip())
    assert major == 3 and minor >= 11

    shutil.copyfile(EXAMPLES / "test-gate" / "gate.py", tmp_path / "gate.py")
    code, stdout, stderr = _run_in_check_sandbox(
        [python, "gate.py", "--check-program", "python3"], tmp_path
    )
    assert code == 0, (stdout, stderr)
    assert stdout == b"" and stderr == b""
