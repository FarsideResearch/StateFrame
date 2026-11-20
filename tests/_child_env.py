"""Environment for child interpreters started by tests.

A child started with ``python -c`` sees neither pytest's rootdir insertion nor the
caller's working directory, so it gets the tests, src and engine directories
explicitly. The caller's PYTHONPATH, if any, follows them.
"""
from __future__ import annotations

import os
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def child_env(base=None):
    env = dict(os.environ if base is None else base)
    paths = [str(TESTS), str(ROOT / "src"), str(ROOT / "src" / "stateframe" / "engine")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
