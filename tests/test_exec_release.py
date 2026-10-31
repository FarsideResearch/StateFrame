import hashlib
import json
from pathlib import Path

from kernel_release_switch_registry import _declared_python_files


def test_execution_modules_are_release_declared_with_current_hashes():
    root = Path(__file__).resolve().parents[1] / "src/stateframe/engine"
    manifest = json.loads((root / "release-manifest.json").read_text())
    declared = {path.name for path in _declared_python_files(root)}
    for name in ("kernel_exec_capability.py", "kernel_exec_session.py", "kernel_exec_supervisor.py",
                 "kernel_exec_tracker.py", "kernel_job_output.py", "sf_launch.py", "kernel_test_isolation.py",
                 "kernel_safe_probe.py"):
        assert name in declared
        assert manifest["files"][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
