"""Choose the interpreter that runs the user's tests.

The engine runs on StateFrame's own interpreter. The repository's tests must run on the
PROJECT's interpreter, or any project with dependencies fails its checks with import errors
that look like a broken task. Order:

1. ``--test-python`` (explicit).
2. ``--python`` when the user passed it (backward compatible: it used to mean engine + tests).
3. A virtual environment inside the repository: ``.venv``, ``venv``, ``env`` (uv and an
   in-project poetry environment both create ``.venv``).
4. ``python3`` then ``python`` on PATH.

StateFrame's own virtual environment is never chosen silently: PATH interpreters that live inside it
are skipped, and when nothing qualifies the run stops with a message naming ``--test-python``.
No external tool (poetry, uv) is run: detection is offline and only looks at files.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

VENV_DIRS = (".venv", "venv", "env")


class TestInterpreterError(ValueError):
    pass


def _venv_python(directory: Path) -> Path | None:
    for name in ("python3", "python"):
        candidate = directory / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _is_own_environment(python: Path, own_prefix: Path) -> bool:
    """True when ``python`` belongs to StateFrame's own virtual environment."""
    if sys.prefix == sys.base_prefix:
        return False
    try:
        parent = python.parent.resolve()
        unresolved = python.parent.absolute()
    except OSError:
        return False
    prefix = own_prefix.resolve()
    return any(p == prefix or prefix in p.parents for p in (parent, unresolved))


def _path_interpreter(name: str, search_path: str, own_prefix: Path) -> str | None:
    """Prefer the first external match; retain an owned match for the failure message."""
    own_candidate: str | None = None
    entries = search_path.split(os.pathsep) if search_path else ()
    for entry in entries:
        candidate = shutil.which(name, path=entry or os.curdir)
        if candidate is None:
            continue
        candidate_path = Path(candidate)
        # Resolve only the parent so ownership and preflight see the same route without following venv links.
        try:
            candidate = str(candidate_path.parent.resolve() / candidate_path.name)
        except (OSError, RuntimeError):
            pass
        if not _is_own_environment(Path(candidate), own_prefix):
            return candidate
        own_candidate = own_candidate or candidate
    return own_candidate


def choose(repo: Path, *, test_python: str | None, python: str | None,
           own_prefix: Path | None = None, path_env: str | None = None) -> tuple[str, str]:
    """Return ``(interpreter, reason)``; the interpreter is a name or path, resolved later."""
    if test_python:
        return test_python, "--test-python"
    if python:
        return python, "--python (engine and tests)"
    root = Path(repo)
    for name in VENV_DIRS:
        found = _venv_python(root / name)
        if found is not None:
            return str(found), f"project environment {name}/"
    own = Path(own_prefix if own_prefix is not None else sys.prefix)
    search_path = path_env if path_env is not None else os.environ.get("PATH", "")
    on_path = shutil.which("python3", path=search_path)
    if on_path and not _is_own_environment(Path(on_path), own):
        return on_path, "python3 on PATH (no project environment found)"
    python_on_path = _path_interpreter("python", search_path, own)
    if python_on_path and not _is_own_environment(Path(python_on_path), own):
        return python_on_path, "python on PATH (no project environment found)"
    reason = "; ".join(
        f"{name} on PATH " + ("is StateFrame's own environment" if candidate else "was not found")
        for name, candidate in (("python3", on_path), ("python", python_on_path))
    )
    raise TestInterpreterError(
        "no interpreter for your tests: the repository has no .venv/, venv/ or env/, and "
        + reason + "; pass --test-python PATH (the Python that has your project's dependencies and pytest)")
