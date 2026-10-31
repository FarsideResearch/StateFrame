import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stateframe" / "engine"))


def pytest_configure(config):
    config.addinivalue_line("markers", "host_sandbox: requires host sandbox or local sockets")
