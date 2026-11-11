"""Mutations OFF never reaches the mutation platform check, even where it would refuse."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_mutate_parity import _remove_run
from tools.compare_mutation_off_requests import capture, write_launcher


CURRENT = Path(__file__).resolve().parents[1]
POISONED = ("kernel_mutate_exec.py", "kernel_mutate_qualify.py", "kernel_mutate_runtime.py")


def _poisoned_tree(root: Path) -> Path:
    """A copy whose mutation platform modules fail on import, so any import with mutations off fails the run."""
    tree = root / "tree"
    tree.mkdir()
    for name in ("src", "examples"):
        shutil.copytree(CURRENT / name, tree / name, ignore=shutil.ignore_patterns("__pycache__"))
    engine = tree / "src" / "stateframe" / "engine"
    for name in POISONED:
        assert (engine / name).is_file()
        (engine / name).write_text("raise RuntimeError('mutation platform imported with mutations off')\n")
    host = tree / "src" / "stateframe" / "mutate_host.py"
    host.write_text("raise RuntimeError('mutation host imported with mutations off')\n")
    return tree


@pytest.mark.host_sandbox
def test_default_run_never_touches_mutation_platform(tmp_path, monkeypatch):
    marker = tmp_path / "ldd-called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ldd = bin_dir / "ldd"
    ldd.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    ldd.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home" / ".stateframe"))
    launcher = tmp_path / "launch.py"
    write_launcher(launcher)
    tree = _poisoned_tree(tmp_path)

    repo, run = tmp_path / "repo", tmp_path / "run"
    poisoned_request, poisoned_tools, spec = capture(tree, sys.executable, launcher, "discount_bug",
                                                     repo, run, stable_tool_head="on")
    assert json.loads((run / "summary.json").read_text())["engine_result"] == "SUBMITTED"
    assert "mutations" not in spec
    assert not marker.exists()
    engine_log = (run / "engine.log").read_text(errors="replace")
    assert "mutations off" not in engine_log and "MUTATION_PLATFORM" not in engine_log

    # Same repo and run paths as the poisoned run, so request bytes are comparable.
    _remove_run(run)
    clean_request, clean_tools, _ = capture(CURRENT, sys.executable, launcher, "discount_bug",
                                            repo, run, stable_tool_head="on")
    assert poisoned_request == clean_request
    print("DEFAULT_REQUEST_SHA256", hashlib.sha256(clean_request).hexdigest(), clean_tools[0][:3])
    assert poisoned_tools == clean_tools


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform == "darwin", reason="Linux only: ldd")
@pytest.mark.skipif(os.environ.get("SF_TEST_LDD_MASKED") != "1",
                    reason="set SF_TEST_LDD_MASKED=1 in a test container whose /usr/bin/ldd is masked")
def test_without_ldd_default_run_is_identical_and_mutations_refuse_naming_ldd(tmp_path, monkeypatch):
    assert not os.access("/usr/bin/ldd", os.X_OK) or os.path.getsize("/usr/bin/ldd") == 0
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home" / ".stateframe"))
    launcher = tmp_path / "launch.py"
    write_launcher(launcher)
    repo, run = tmp_path / "repo", tmp_path / "run"
    request, tools, spec = capture(CURRENT, sys.executable, launcher, "discount_bug", repo, run,
                                   stable_tool_head="on")
    assert json.loads((run / "summary.json").read_text())["engine_result"] == "SUBMITTED"
    assert "mutations" not in spec
    print("NO_LDD_REQUEST_SHA256", hashlib.sha256(request).hexdigest(), tools[0][:3])
    result = subprocess.run([sys.executable, "-m", "stateframe", "mutations", "selftest", "--json"],
                            env={**os.environ, "PYTHONPATH": str(CURRENT / "src")},
                            capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert "ldd is unavailable" in result.stdout + result.stderr
