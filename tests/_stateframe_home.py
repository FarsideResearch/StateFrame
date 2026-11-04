"""The STATEFRAME_HOME a test subprocess must get: a temporary one, never the developer's real ~/.stateframe.

Subprocess helpers drop inherited STATEFRAME_* switches so a run sees the packaged profile. That also dropped
STATEFRAME_HOME, so every engine launch from them fell back to ~/.stateframe and left a new env seed there; on
11-03 the shared Mac collected 52 GB of them in a day. Every helper that strips STATEFRAME_* must put this back
(tests/test_stateframe_home_guard.py enforces it).
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_fallback: str | None = None


def _real_home() -> Path:
    return (Path.home() / ".stateframe").expanduser().resolve()


def isolated_stateframe_home() -> str:
    """The session's temporary home (set by tests/conftest.py), or a private one; never the real home."""
    global _fallback
    current = os.environ.get("STATEFRAME_HOME")
    if current and Path(current).expanduser().resolve() != _real_home():
        return current
    if _fallback is None:
        _fallback = tempfile.mkdtemp(prefix="sf-test-home-")
        atexit.register(shutil.rmtree, _fallback, True)
    return _fallback
