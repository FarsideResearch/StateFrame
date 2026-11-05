"""Shipped check selection keeps the caller's module state and oracle contracts."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

from probes import react_repair
from stateframe.engine import check_selection


def test_repair_module_identity_and_mc_selector_overlay(monkeypatch, tmp_path: Path):
    assert check_selection.rr is react_repair
    mc_root = tmp_path / "mc"
    mc_root.mkdir()
    monkeypatch.setitem(react_repair.REPOS, "mc", mc_root / "backend")
    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if "diff" in argv:
            return SimpleNamespace(returncode=0, stdout=(
                "+++ b/backend/tests/test_one.py\n"
                "+def test_fixed_behavior():\n"
                "+    assert True\n"))
        if "--name-only" in argv:
            return SimpleNamespace(returncode=0, stdout="backend/tests/test_one.py\n")
        return SimpleNamespace(returncode=0, stdout="def test_fixed_behavior(): assert True\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    row = {"repo": "mc", "sha": "abc123", "fn": "unique_mc_selector_case"}
    assert check_selection.fail_to_pass_ids(row) == ["tests/test_one.py::test_fixed_behavior"]
    assert check_selection.overlay_fix_tests(tmp_path / "grading", row) == 1
    assert (tmp_path / "grading/tests/test_one.py").read_text() == (
        "def test_fixed_behavior(): assert True\n")
    assert all(str(mc_root) in str(argv) for argv in calls)


def test_multifile_selectors_and_pristine_overlay(tmp_path: Path):
    row = {
        "files": {"app.py": {"parent_src": "x = 1\n"}, "util.py": {"parent_src": "y = 2\n"}},
        "oracle": {"selectors": ["tests/test_a.py::test_one", "tests/test_a.py::test_one", "tests/test_b.py"],
                   "test_files": {"tests/test_a.py": "def test_one(): assert True\n"}},
    }
    assert check_selection.fail_to_pass_ids(row) == ["tests/test_a.py::test_one", "tests/test_b.py"]
    assert check_selection.overlay_fix_tests(tmp_path, row) == 1
    assert (tmp_path / "tests/test_a.py").read_text() == "def test_one(): assert True\n"


def test_multifile_overlay_rejects_path_escape(tmp_path: Path):
    row = {"files": {"app.py": {}}, "oracle": {"test_files": {"../outside.py": "x=1"}}}
    try:
        check_selection.overlay_fix_tests(tmp_path, row)
    except ValueError as error:
        assert "escapes checkout" in str(error)
    else:
        raise AssertionError("unsafe oracle path accepted")
