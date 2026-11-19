"""Exercise the engine imports from the installed wheel."""

import subprocess
import sys
from pathlib import Path

from _timeouts import scaled

ROOT = Path(__file__).resolve().parents[1]
IMPORT_ENGINE = """
import importlib
import sys
import sysconfig
from pathlib import Path

engine = Path(sysconfig.get_path("purelib")) / "stateframe" / "engine"
files = sorted(engine.rglob("*.py"))
assert files, f"No installed engine modules in {engine}"

# The engine child exposes this directory for its top-level kernel imports.
sys.path.insert(0, str(engine))
for path in files:
    parts = path.relative_to(engine).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    name = ".".join(parts) or "stateframe.engine"
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        error.add_note(f"While importing installed engine module {name}")
        raise
    assert Path(module.__file__).resolve() == path.resolve(), (name, module.__file__, path)
print("Installed engine imports passed.")
"""


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=scaled(120), check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_wheel_imports(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    _run([sys.executable, "-I", "-m", "venv", str(venv)], tmp_path)
    python = str(venv / "bin" / "python")
    _run([
        python, "-I", "-m", "pip", "--isolated", "install", "--no-index",
        "--no-build-isolation", "--no-deps", "--no-cache-dir", "--no-compile",
        "--disable-pip-version-check", str(ROOT),
    ], tmp_path)
    result = _run([python, "-I", "-B", "-c", IMPORT_ENGINE], tmp_path)
    assert "Installed engine imports passed." in result.stdout.splitlines(), (
        result.stdout + result.stderr
    )
