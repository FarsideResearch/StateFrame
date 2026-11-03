"""Repository tests import the checkout's product package and engine modules."""
import sys
import importlib.metadata
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Fixture repositories under tests/e2e carry their own tests; they run inside engine runs only.
collect_ignore_glob = ["e2e/*", "fixtures/*"]  # recorded repo copies in fixtures/ are data, never tests
for path in (ROOT / "src", ROOT / "src" / "stateframe" / "engine"):
    sys.path.insert(0, str(path))


@pytest.fixture(autouse=True, scope="session")
def _isolated_stateframe_home(tmp_path_factory):
    """Runs under test build env seeds and user state; keep them out of the real ~/.stateframe."""
    import os

    if os.environ.get("STATEFRAME_HOME"):
        yield
        return
    os.environ["STATEFRAME_HOME"] = str(tmp_path_factory.mktemp("stateframe-home"))
    try:
        yield
    finally:
        os.environ.pop("STATEFRAME_HOME", None)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_sandbox: starts an OS sandbox (sandbox-exec/bwrap) or a local socket service")


@pytest.fixture(autouse=True)
def _restore_transport_sender():
    """_child.install_wire_recorder replaces kernel_bounded_http.request_bytes for the whole
    process. Put the real sender back after each test so a recorder installed by one test
    (for example a routed one) never intercepts a later test's sends."""
    import kernel_bounded_http

    original = kernel_bounded_http.request_bytes
    yield
    kernel_bounded_http.request_bytes = original


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
