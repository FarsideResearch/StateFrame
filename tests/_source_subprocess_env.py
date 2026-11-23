"""Explicit import paths for subprocesses that exercise this source checkout."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = (
    str(ROOT / "src"),
    str(ROOT / "src" / "stateframe" / "engine"),
)


def source_subprocess_env(*, inherit_pythonpath: bool = False) -> dict[str, str]:
    """Use checkout sources first, optionally followed by the caller's dependency paths."""
    env = os.environ.copy()
    paths = _SOURCE_PATHS
    if inherit_pythonpath:
        # The child can start in a different cwd from the caller.
        inherited = env.get("PYTHONPATH", "").split(os.pathsep)
        paths += tuple(os.path.abspath(path) for path in inherited if path)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env
