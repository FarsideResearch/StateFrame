"""Keep direct event-journal readers behind the storage shim."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
READ_CALLS = {"open", "read_text", "read_bytes", "glob", "rglob"}
SHIM_FILES = {"kernel_measurement.py", "kernel_journal_store.py"}


def raw_journal_reads(root: Path) -> list[str]:
    violations = []
    for folder in (root / "src", root / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            if path.name in SHIM_FILES:
                continue
            source = path.read_text(encoding="utf-8")
            if "events.jsonl" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else (
                    call.func.id if isinstance(call.func, ast.Name) else "")
                if name not in READ_CALLS:
                    continue
                target = call.func.value if isinstance(call.func, ast.Attribute) else call
                expressions = [target, *call.args]
                if not any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                           and "events.jsonl" in node.value
                           for expression in expressions for node in ast.walk(expression)):
                    continue
                if (path.relative_to(root) == Path("src/stateframe/resume.py")
                        and name == "open" and any(isinstance(node, ast.Constant)
                                                    and node.value == "a"
                                                    for arg in call.args for node in ast.walk(arg))):
                    # Archive-attempt diagnostics use a separate checkpoint journal.
                    continue
                violations.append(f"{path.relative_to(root)}:{call.lineno}")
    return violations


def test_all_event_journal_readers_use_shim():
    assert raw_journal_reads(ROOT) == []


def test_planted_direct_read_is_caught(tmp_path):
    source = tmp_path / "src" / "fake.py"
    source.parent.mkdir()
    source.write_text('from pathlib import Path\nPath("events.jsonl").read_text()\n'
                      'Path(".").glob("*/events.jsonl")\n'
                      'Path("events.jsonl").open("a")\n')
    assert raw_journal_reads(tmp_path) == ["src/fake.py:2", "src/fake.py:3", "src/fake.py:4"]


def test_storage_module_imports_without_measurement_or_path_side_effects():
    script = """
import importlib
import os
import sys

before = list(sys.path)
importlib.import_module("stateframe.engine.kernel_journal_store")
assert sys.path == before
assert "kernel_measurement" not in sys.modules
assert "stateframe.engine.kernel_measurement" not in sys.modules

sys.path.insert(0, os.environ["STATEFRAME_ENGINE_DIR"])
before = list(sys.path)
importlib.import_module("kernel_journal_store")
assert sys.path == before
assert "kernel_measurement" not in sys.modules
assert "stateframe.engine.kernel_measurement" not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["STATEFRAME_ENGINE_DIR"] = str(ROOT / "src" / "stateframe" / "engine")
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environment,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
