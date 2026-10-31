"""Repository tests import the checkout's product package and engine modules."""
import sys
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Fixture repositories under tests/e2e carry their own tests; they run inside engine runs only.
collect_ignore_glob = ["e2e/*"]
for path in (ROOT / "src", ROOT / "src" / "stateframe" / "engine"):
    sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_sandbox: starts an OS sandbox (sandbox-exec/bwrap) or a local socket service")


@pytest.fixture
def clean_test_environment(tmp_path):
    """Real pytest and pip, with no ambient editable installs or startup hooks."""
    def create(destination):
        subprocess.run([sys.executable, "-I", "-m", "venv", str(destination)],
                       check=True, capture_output=True, timeout=60)
        site = next(destination.glob("lib/python*/site-packages"))
        # Copy only installed distribution files. In particular, do not inherit
        # the host's editable StateFrame .pth or grant its source tree authority.
        for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments"):
            distribution = importlib.metadata.distribution(name)
            for relative in distribution.files:
                if ".." in relative.parts or relative.suffix == ".pyc":
                    continue
                source = Path(distribution.locate_file(relative))
                if source.is_file():
                    target = site / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        return destination / "bin/python"
    return create
