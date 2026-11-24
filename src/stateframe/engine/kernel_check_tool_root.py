"""The install root of a package manager named by the user's own test command.

Only Linux checks use this. A root is returned for a read-only mount when every rule holds;
anything uncertain is refused with a reason and nothing is mounted.
"""
from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .kernel_safe_paths import safe_home
    from .kernel_secret_paths import is_secret_file
else:
    from kernel_safe_paths import safe_home
    from kernel_secret_paths import is_secret_file

PACKAGE_TOOLS = frozenset({"npm", "npx", "pnpm", "yarn"})
_SYSTEM = tuple(map(Path, ("/usr", "/bin", "/sbin", "/lib", "/lib64")))
_PRIVATE_NAMES = frozenset({
    ".config", ".cache", ".ssh", ".gnupg", ".aws", ".azure", ".gcloud", ".kube", ".docker",
    ".npm", ".yarn", ".pki", ".password-store", ".secrets", ".env", "keyrings", "credentials",
})


# The host's record for this engine process. It is never part of the task row, the journal or a
# request; only the check launcher reads it.
_recorded_root: str | None = None


def record_root(record: object) -> None:
    """Keep the root from the run's host-written spec, for this process only."""
    global _recorded_root
    root = record.get("root") if isinstance(record, Mapping) else None
    _recorded_root = root if isinstance(root, str) and os.path.isabs(root) else None


def recorded_root() -> str | None:
    return _recorded_root


@dataclass(frozen=True)
class ToolRoot:
    program: str
    entry: str       # the PATH entry that names the tool
    directory: str   # the resolved directory that holds it
    root: str        # the resolved install root

    def record(self) -> dict[str, str]:
        return {"program": self.program, "root": self.root}


def _homes(environ: Mapping[str, str]) -> set[Path]:
    homes: set[Path] = set()
    # The second lookup has no HOME key, so it names the account's own home.
    for value in (safe_home(environ), safe_home({})):
        if value is not None and value.is_absolute():
            homes.update((Path(os.path.normpath(value)), value.resolve()))
    return homes


def _trusted(path: Path) -> str | None:
    """Why this exact file or directory cannot be trusted, if so."""
    status = path.stat()
    if status.st_uid not in (0, os.getuid()):
        return f"{path} is not owned by this user or root"
    if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{path} is writable by its group or by everyone"
    return None


def _root_refusal(root: Path, homes: set[Path]) -> str | None:
    if len(root.parts) <= 2:
        return f"{root} is a top-level directory"
    for home in homes:
        if root == home or home.is_relative_to(root):
            return f"{root} is or contains the home directory"
        if root.parent == home and root.name.startswith("."):
            return f"{root} is a private directory in the home directory"
    if any(part.casefold() in _PRIVATE_NAMES or is_secret_file(part) for part in root.parts):
        return f"{root} is inside a configuration or credential directory"
    for parent in root.parents:
        status = parent.stat()
        if status.st_uid not in (0, os.getuid()):
            return f"{parent} is not owned by this user or root"
        if status.st_mode & stat.S_IWOTH and not status.st_mode & stat.S_ISVTX:
            return f"{parent} is writable by everyone"
    for directory in (root, root / "etc"):
        if directory.is_dir():
            for entry in os.scandir(directory):
                if entry.name.casefold() in _PRIVATE_NAMES | {"npmrc"} or is_secret_file(entry.name):
                    return f"{directory} holds a configuration or credential file: {entry.name}"
    return None


def find_tool_root(program: str, path: str, workspace: Path, *,
                   environ: Mapping[str, str] | None = None) -> tuple[ToolRoot | None, str | None]:
    """Return (root, None), (None, reason) when refused, or (None, None) when not applicable."""
    if program not in PACKAGE_TOOLS:
        return None, None
    entries = path.split(os.pathsep)
    if not all(os.path.isabs(entry) for entry in entries):
        return None, "PATH has a relative entry"
    found = shutil.which(program, path=path)
    if found is None:
        return None, None
    try:
        entry = Path(found).parent
        directory = entry.resolve(strict=True)
        if any(directory.is_relative_to(system) for system in _SYSTEM):
            return None, None
        tool = Path(found).resolve(strict=True)
        root = directory.parent
        workspace = workspace.resolve()
        if root.is_relative_to(workspace) or workspace.is_relative_to(root):
            return None, f"{root} overlaps the checked repository"
        if "node_modules" in root.parts:
            return None, f"{root} is inside a node_modules directory"
        # Only the layout of a Node installation: ROOT/bin/TOOL, ROOT/bin/node, ROOT/lib/node_modules.
        node = directory / "node"
        modules = root / "lib" / "node_modules"
        if (directory.name != "bin" or not node.is_file() or not os.access(node, os.X_OK)
                or not modules.is_dir()):
            return None, f"{root} is not a Node installation"
        resolved_node = node.resolve(strict=True)
        for resolved in (tool, resolved_node, modules.resolve(strict=True)):
            if not resolved.is_relative_to(root):
                return None, f"{resolved} is outside the install root {root}"
        refusal = _root_refusal(root, _homes(os.environ if environ is None else environ))
        if refusal is not None:
            return None, refusal
        for item in dict.fromkeys((root, directory, modules.parent, modules, tool.parent, tool,
                                   resolved_node.parent, resolved_node)):
            refusal = _trusted(item)
            if refusal is not None:
                return None, refusal
    except (OSError, RuntimeError) as error:
        return None, f"{program} could not be checked ({type(error).__name__})"
    return ToolRoot(program, str(entry), str(directory), str(root)), None


def sandbox_path(path: str, selected: ToolRoot) -> str:
    """Name the mounted directory itself, so symlinked PATH entries need no extra mount."""
    return os.pathsep.join(selected.directory if os.path.normpath(entry) == selected.entry else entry
                           for entry in path.split(os.pathsep))
