"""Local source imports and explicit host-boundary acceptance marker."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/stateframe/engine")]


def pytest_configure(config):
    config.addinivalue_line("markers", "host_sandbox: needs OS sandbox or sockets outside the outer seatbelt")
