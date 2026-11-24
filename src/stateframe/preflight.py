"""Small, standard-library preflight checks shared by StateFrame entry points."""
from __future__ import annotations

import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .engine.kernel_safe_paths import safe_expanduser

class PreflightError(RuntimeError):
    """A required local interpreter or sandbox prerequisite is unavailable."""


def resolve_interpreter(path_or_name: str | os.PathLike[str], cwd: str | os.PathLike[str]) -> str:
    """Resolve an executable name or caller-relative path to an absolute file path."""
    raw = os.fspath(path_or_name)
    if not raw or not raw.strip():
        raise PreflightError("Python interpreter path or command name is empty")

    try:
        caller_cwd = safe_expanduser(Path(cwd)).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PreflightError(f"caller working directory is unavailable: {cwd}") from error
    if not caller_cwd.is_dir():
        raise PreflightError(f"caller working directory is not a directory: {caller_cwd}")

    candidate_path = safe_expanduser(Path(raw))
    has_path_component = candidate_path.is_absolute() or os.sep in raw or (os.altsep and os.altsep in raw)
    if has_path_component:
        candidate = candidate_path if candidate_path.is_absolute() else caller_cwd / candidate_path
        candidates = (candidate,)
    else:
        search_path: list[str] = []
        for directory in os.get_exec_path():
            item = Path(directory or caller_cwd)
            if not item.is_absolute():
                item = caller_cwd / item
            search_path.append(str(item))
        found = shutil.which(raw, path=os.pathsep.join(search_path))
        if found is None:
            raise PreflightError(f"Python interpreter not found on PATH: {raw}")
        candidates = (Path(found),)

    # Absolute but NOT symlink-resolved: a venv's bin/python is a symlink to the base interpreter, and
    # running the resolved target would leave the venv (losing its site-packages, pytest included).
    candidate = Path(os.path.abspath(candidates[0]))
    try:
        target = candidate.resolve(strict=True)  # only to validate that the link chain ends in a real file
    except (OSError, RuntimeError) as error:
        raise PreflightError(f"Python interpreter does not exist: {candidate}") from error
    if not target.is_file():
        raise PreflightError(f"Python interpreter is not a file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise PreflightError(f"Python interpreter is not executable: {candidate}")
    return str(candidate)


def check_python_version(
    min: tuple[int, int] = (3, 11),
    *,
    version: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Return the checked major/minor version or raise a readable preflight error."""
    required = tuple(min)
    if len(required) != 2 or any(not isinstance(part, int) or part < 0 for part in required):
        raise ValueError("min must be a (major, minor) integer pair")
    current = tuple(sys.version_info[:2] if version is None else version[:2])
    if len(current) != 2 or any(not isinstance(part, int) or part < 0 for part in current):
        raise ValueError("version must provide a (major, minor) integer pair")
    if current < required:
        raise PreflightError(
            f"Python {required[0]}.{required[1]} or newer is required; found Python {current[0]}.{current[1]}"
        )
    full = tuple(sys.version_info[:3] if version is None else version[:3])
    if full in EXCLUDED_PYTHON_VERSIONS:
        raise PreflightError(f"Python {'.'.join(map(str, full))} is not supported: {EXCLUDED_PYTHON_VERSIONS[full]}")
    return current


# Exact patch releases a required pinned dependency refuses (its Requires-Python), with the reason shown to the user.
EXCLUDED_PYTHON_VERSIONS = {
    (3, 14, 1): "networkx 3.6.1, a required dependency, excludes this release; use another 3.14.x or Python 3.11-3.13",
}


def sandbox_available() -> bool:
    """Report whether this platform's required command sandbox is on PATH."""
    system = platform.system()
    if system == "Darwin":
        command = "sandbox-exec"
    elif system == "Linux":
        command = "bwrap"
    else:
        return False
    return shutil.which(command) is not None


MISSING_PROGRAM_EXIT = 113
_SANDBOX_VISIBLE = ("/usr", "/bin", "/lib", "/lib64")


def check_tool_root(command: str, *, test_env: Mapping[str, str] | None, repo: Path,
                    user_test_command: bool):
    """The package-manager install root a check mounts, and why one was refused."""
    if not user_test_command or not sys.platform.startswith("linux"):
        return None, None
    try:
        arguments = shlex.split(command)
    except ValueError:
        return None, None
    if not arguments or arguments[0] not in {"npm", "npx", "pnpm", "yarn"}:
        return None, None
    from .engine.kernel_check_tool_root import find_tool_root
    path = (test_env or {}).get("PATH", os.environ.get("PATH", os.defpath))
    return find_tool_root(arguments[0], path, repo)


def missing_program_message(command: str, *, test_env: Mapping[str, str] | None,
                            repo: Path, user_test_command: bool,
                            run_probe: Callable[[str, Path], Mapping[str, object]]) -> str | None:
    """Return a refusal only when the check sandbox itself reports the program as absent.

    Names the sandbox certainly sees are answered without a probe. Anything uncertain, such as
    a relative PATH entry or a probe that could not run, is left to the check itself.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        arguments = shlex.split(command)
    except ValueError:
        return None
    program = arguments[0] if arguments else ""
    if not program or "/" in program or "\0" in program or program.startswith("-"):
        return None
    if command.startswith("python "):
        return None  # The runner substitutes the selected interpreter.
    path = (test_env or {}).get("PATH", os.environ.get("PATH", os.defpath))
    if not all(os.path.isabs(entry) for entry in path.split(os.pathsep)):
        return None
    found = shutil.which(program, path=path)
    if found is not None and Path(found).is_relative_to(sys.prefix):
        return None
    if found is not None and any(
            Path(found).is_relative_to(root) and Path(found).resolve().is_relative_to("/usr")
            for root in _SANDBOX_VISIBLE):
        return None
    tool_root, refusal = check_tool_root(command, test_env=test_env, repo=repo,
                                         user_test_command=user_test_command)
    if tool_root is not None:
        return None
    probe = shlex.join(["sh", "-c", f'command -v "$0" >/dev/null || exit {MISSING_PROGRAM_EXIT}', program])
    try:
        with tempfile.TemporaryDirectory(prefix="stateframe-program-") as directory:
            empty = Path(directory) / "repo"
            empty.mkdir()
            result = run_probe(probe, empty)
    except (RuntimeError, OSError, ValueError):
        return None
    if result.get("exit_code") != MISSING_PROGRAM_EXIT:
        return None
    if found is None:
        return (f"check command not found: {program}; install it or change the test command")
    example = shlex.join([found, *arguments[1:]])
    reason = f"; not mounted: {refusal}" if refusal else ""
    return (f"check command not found in the check sandbox: {program} (installed at {found}{reason}); "
            f"install it system-wide or try its absolute path: --test-cmd {shlex.quote(example)}")
