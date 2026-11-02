"""Installed release examples must reach a model request and submit."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .doc_command_sweep import commands, outcome, sweep


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.host_sandbox


def test_unmapped_documented_example_fails_collection(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "README.md").write_text("# Example\n", encoding="utf-8")
    (tmp_path / "QUICKSTART.md").write_text(
        "```bash\nstateframe run --repo examples/missing --task 'Build it'\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="no bundled script"):
        commands(tmp_path)


def test_unmapped_documented_repository_fails_collection(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "README.md").write_text(
        "```bash\nstateframe run --repo ~/code/another --task 'Build it'\n```\n",
        encoding="utf-8",
    )
    (tmp_path / "QUICKSTART.md").write_text("# Quickstart\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="unmapped run command"):
        commands(tmp_path)


def test_documented_and_bundled_examples_submit_from_clean_install(tmp_path):
    cases = commands(ROOT)
    assert {case.example for case in cases} == {
        path.name.removesuffix(".script.json")
        for path in (ROOT / "examples").glob("*.script.json")
    }
    results = sweep(ROOT, tmp_path)
    assert len(results) == len(cases)
    assert all(outcome(requests, result) for _, requests, result in results), results


def test_doc_sweep_rejects_broken_release_before_first_request(tmp_path):
    """The released failure is a positive control for the full installed sweep."""
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", "f848c2e"],
        capture_output=True, check=True,
    ).stdout
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["tar", "-xf", "-", "-C", str(source)], input=archive, check=True)
    case = next(case for case in commands(source) if case.example == "tiny_site")
    results = sweep(source, tmp_path / "run", cases=(case,))
    assert len(results) == 1
    assert not outcome(results[0][1], results[0][2]), results
    assert results[0][1] == 0
