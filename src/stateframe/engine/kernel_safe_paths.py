"""Safe home and tilde-path handling for host and engine code."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

try:
    import pwd
except ImportError:  # Windows has no passwd database.
    pwd = None


def container_active() -> bool:
    """Check either boundary import identity without importing the boundary."""
    import sys

    for name in ("stateframe.engine.kernel_execution_boundary", "kernel_execution_boundary"):
        boundary = sys.modules.get(name)
        active = getattr(boundary, "active", None)
        if callable(active) and getattr(active(), "backend", None) == "container":
            return True
    return False


def safe_home(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the selected home directory, or ``None`` when the uid has no home.

    A supplied environment is authoritative: its ``HOME`` is used when present,
    and an empty value means no home. Otherwise the process environment is used.
    If that source has no ``HOME`` key, consult the passwd database when available.
    """
    try:
        source = env if env is not None else os.environ
        configured = source.get("HOME")
        if configured is not None:
            return Path(configured) if configured else None
        if pwd is None or not hasattr(os, "getuid"):
            return None
        directory = pwd.getpwuid(os.getuid()).pw_dir
        return Path(directory) if directory else None
    except (RuntimeError, KeyError, OSError):
        return None


def safe_expanduser(path: str | os.PathLike[str],
                    env: Mapping[str, str] | None = None) -> Path:
    """Expand a current-user or named-user tilde path without calling ``expanduser``.

    Non-tilde paths are returned unchanged. A tilde path with no resolvable home
    raises a clear RuntimeError rather than depending on ``Path.home`` behavior.
    """
    candidate = Path(path)
    raw = os.fspath(candidate)
    if not raw.startswith("~"):
        return candidate

    username, separator, suffix = raw.partition(os.sep)
    if username == "~":
        home = safe_home(env)
    elif pwd is not None:
        try:
            directory = pwd.getpwnam(username[1:]).pw_dir
        except (RuntimeError, KeyError, OSError):
            directory = ""
        home = Path(directory) if directory else None
    else:
        home = None
    if home is None:
        raise RuntimeError("Could not determine home directory.")
    return home / suffix if separator and suffix else home
