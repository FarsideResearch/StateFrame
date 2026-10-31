from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stateframe" / "engine"))


def pytest_configure(config):
    config.addinivalue_line("markers", "host_sandbox: requires local sockets or a host sandbox executable")
