"""Repository tests use the checkout's product and engine modules."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
collect_ignore_glob = ["e2e/*"]
for path in (ROOT / "src", ROOT / "src" / "stateframe" / "engine"):
    sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_sandbox: starts the engine sandbox or a socket service")
