"""Source-checkout imports must survive a fresh child interpreter and foreign cwd."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _source_subprocess_env import source_subprocess_env


ROOT = Path(__file__).resolve().parents[1]
_CHILD_IMPORT_AND_INTERRUPT = """
import json
from pathlib import Path
from stateframe import resume
import kernel_continuation_journal

try:
    raise RuntimeError("intentional child interruption")
except RuntimeError as error:
    assert str(error) == "intentional child interruption"
    print(json.dumps({
        "reached": str(error),
        "stateframe": str(Path(resume.__file__).resolve()),
        "journal": str(Path(kernel_continuation_journal.__file__).resolve()),
    }))
"""


def test_child_imports_checkout_modules_before_interruption(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated-pythonpath"
    unrelated.mkdir()
    (unrelated / "kernel_continuation_journal.py").write_text(
        'raise RuntimeError("unrelated PYTHONPATH won")\n', encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(unrelated))
    child_env = source_subprocess_env()

    child = subprocess.run(
        [sys.executable, "-c", _CHILD_IMPORT_AND_INTERRUPT],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["reached"] == "intentional child interruption"
    assert Path(result["stateframe"]).is_relative_to(ROOT / "src" / "stateframe")
    assert Path(result["journal"]).is_relative_to(ROOT / "src" / "stateframe" / "engine")
    assert child_env["PYTHONPATH"].split(os.pathsep) == [
        str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine")
    ]


def test_child_can_import_pytest_from_the_inherited_dependency_path(tmp_path, monkeypatch):
    dependencies = tmp_path / "test dependencies"
    dependencies.mkdir()
    (dependencies / "pytest.py").write_text("PATH_ONLY = True\n", encoding="utf-8")
    (dependencies / "kernel_continuation_journal.py").write_text(
        'raise RuntimeError("inherited PYTHONPATH won")\n', encoding="utf-8",
    )
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", dependencies.name)

    child = subprocess.run(
        [sys.executable, "-S", "-c", "import pytest, kernel_continuation_journal; assert pytest.PATH_ONLY"],
        cwd=child_cwd, env=source_subprocess_env(inherit_pythonpath=True),
        capture_output=True, text=True, check=False,
    )

    assert child.returncode == 0, child.stderr
