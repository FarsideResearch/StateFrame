"""Repository tests import the checkout's product package and engine modules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Fixture repositories under tests/e2e carry their own tests; they run inside engine runs only.
collect_ignore_glob = ["e2e/*"]
for path in (ROOT / "src", ROOT / "src" / "stateframe" / "engine"):
    sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "host_sandbox: starts an OS sandbox (sandbox-exec/bwrap) or a local socket service")
