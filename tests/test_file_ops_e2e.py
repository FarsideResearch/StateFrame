"""Offline scripted runs exercise the actual CLI, request menu, checks and apply."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "e2e"


def _run(tmp_path, fixture, script, *, edit=()):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / fixture, repo, symlinks=True)
    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine")))
    endpoint = subprocess.Popen(
        [sys.executable, "-m", "stateframe.scripted_endpoint", "--script",
         str(FIXTURES / script), "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        assert url.startswith("http://127.0.0.1:") and url.endswith("/v1"), url
        command = [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Apply the requested module housekeeping and verify imports.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", sys.executable, "--max-turns", "16", "--in-place",
            "--quiet", *edit,
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=environment,
                                timeout=240, check=False)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert result.returncode == 0, result.stdout + result.stderr + str(summary)
        assert summary["succeeded"] is True, summary
        return repo, run_dir, summary
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)


@pytest.mark.host_sandbox
def test_delete_stale_module_and_apply(tmp_path):
    repo, run_dir, summary = _run(tmp_path, "delete_repo", "delete.script.json")
    assert not (repo / "pkg" / "stale.py").exists()
    assert "from pkg import stale" not in (repo / "pkg" / "active.py").read_text()
    assert "pkg/stale.py" in summary["applied_in_place"]
    assert "pkg/stale.py" in (run_dir / "final.diff").read_text()


@pytest.mark.host_sandbox
def test_move_module_and_apply(tmp_path):
    repo, run_dir, summary = _run(tmp_path, "move_repo", "move.script.json")
    assert not (repo / "pkg" / "old.py").exists()
    assert (repo / "pkg" / "new.py").read_text() == "VALUE = 42\n"
    assert "from pkg.new import VALUE" in (repo / "pkg" / "consumer.py").read_text()
    assert {"pkg/old.py", "pkg/new.py"} <= set(summary["applied_in_place"])
    assert "pkg/old.py" in (run_dir / "final.diff").read_text()


@pytest.mark.host_sandbox
def test_deletion_only_submission(tmp_path):
    repo, _, summary = _run(tmp_path, "delete_only_repo", "delete_only.script.json")
    assert not (repo / "pkg" / "stale.py").exists()
    assert summary["applied_in_place"] == ["pkg/stale.py"]


@pytest.mark.host_sandbox
def test_rename_only_submission(tmp_path):
    repo, _, summary = _run(tmp_path, "rename_only_repo", "rename_only.script.json")
    assert not (repo / "pkg" / "old.py").exists()
    assert (repo / "pkg" / "new.py").read_text() == "VALUE = 42\n"
    assert {"pkg/old.py", "pkg/new.py"} <= set(summary["applied_in_place"])


@pytest.mark.host_sandbox
def test_refusals_are_typed_and_run_continues(tmp_path):
    repo, run_dir, summary = _run(
        tmp_path, "move_repo", "refuse.script.json", edit=("--edit", "pkg/consumer.py"))
    assert (repo / "pkg" / "old.py").exists()
    assert not (tmp_path / "escaped.py").exists()
    assert summary["files_changed"] == []
    # The refusals must reach the model: check the recorded request bodies, not a debug log.
    log = "\n".join(path.read_text(errors="replace")
                    for path in sorted((run_dir / "requests").glob("*.json")))
    for code in ("PROTECTED_TEST", "PATH_NOT_FOUND", "OUTSIDE_WORKSPACE",
                 "DESTINATION_EXISTS", "NOT_EDITABLE"):
        assert f"REFUSED {code}:" in log


@pytest.mark.host_sandbox
def test_existing_discount_demo(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    script = ROOT / "examples" / "discount_bug.script.json"
    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine")))
    endpoint = subprocess.Popen(
        [sys.executable, "-m", "stateframe.scripted_endpoint", "--script", str(script),
         "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment)
    try:
        assert endpoint.stdout is not None
        url = endpoint.stdout.readline().strip()
        result = subprocess.run([
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", sys.executable, "--max-turns", "16", "--quiet",
        ], capture_output=True, text=True, env=environment, timeout=240, check=False)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert result.returncode == 0, result.stdout + result.stderr + str(summary)
        assert summary["succeeded"] is True
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)
