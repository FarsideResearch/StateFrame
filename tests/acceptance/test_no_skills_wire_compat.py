"""Requests without selected skills retain the 0.2.1 wire contract."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from stateframe.replay_tape import first_difference, normalized_digest
from ._support import Endpoint, PYTHON, ROOT, copy_repo, environment


pytestmark = pytest.mark.host_sandbox


def _run_tree(root: Path, case: Path, repo: Path, script: Path) -> tuple[bytes, ...]:
    case.mkdir()
    run_dir = case / "run"
    env = environment()
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "src/stateframe/engine")))
    with Endpoint(script) as endpoint:
        result = subprocess.run(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Change app value and verify it", "--endpoint", endpoint.url,
             "--model", "scripted", "--run-dir", str(run_dir), "--test-python", PYTHON,
             "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "8",
             "--shell", "off", "--quiet"], capture_output=True, text=True,
            env=env, timeout=240, check=False)
    requests = tuple(path.read_bytes() for path in sorted((run_dir / "wire").glob("*.request.json")))
    assert result.returncode == 0 and requests, result.stdout + result.stderr
    return requests


def test_no_skills_wire_matches_021(tmp_path):
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "97cd52d", "src"],
        capture_output=True, check=True)
    baseline = tmp_path / "baseline-tree"
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as contents:
        for member in contents:
            target = baseline / member.name
            assert target.resolve().is_relative_to(baseline) and not member.issym() and not member.islnk()
            contents.extract(member, baseline)
    script = tmp_path / "case.script.json"
    script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "str_replace", "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                                               "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "arguments": {"summary": "Changed the app value."}},
    ]}))
    repo_root = tmp_path / "shared"
    repo_root.mkdir()
    repo = copy_repo(repo_root)
    old = _run_tree(baseline, tmp_path / "old", repo, script)
    current = _run_tree(ROOT, tmp_path / "current", repo, script)
    assert len(current) == len(old)
    for number, (expected, actual) in enumerate(zip(old, current), 1):
        assert normalized_digest(actual) == normalized_digest(expected), (
            number, first_difference(expected, actual))
