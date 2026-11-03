"""Private directories owned by this StateFrame installation."""
from __future__ import annotations

import os
import stat
from pathlib import Path


def _create_missing_parents(parent: Path) -> None:
    missing = []
    current = parent
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            raise FileNotFoundError(f"state directory parent is unavailable: {parent}")
        current = current.parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        else:
            os.chmod(directory, 0o700, follow_symlinks=False)


def private_directory(root: Path, *children: str) -> Path:
    """Create or tighten an owned directory without following its final path components.

    The caller supplies the installation root (normally ``~/.stateframe``) and simple
    child names. A foreign, linked, or non-directory component is never repaired.
    """
    if any(name in {"", ".", ".."} or "/" in name for name in children):
        raise ValueError("state directory child must be a simple name")
    _create_missing_parents(root.parent)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    path = root
    try:
        for name in (None, *children):
            if name is not None:
                path = path / name
            created = False
            try:
                if descriptor is None:
                    os.mkdir(path, 0o700)
                else:
                    os.mkdir(name, 0o700, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                # A restrictive umask can make a new directory unreadable even to its owner.
                # Set its mode before opening, then validate and fchmod the opened inode below.
                if descriptor is None:
                    os.chmod(path, 0o700, follow_symlinks=False)
                else:
                    os.chmod(name, 0o700, dir_fd=descriptor, follow_symlinks=False)
            opened = (os.open(path, flags) if descriptor is None else
                      os.open(name, flags, dir_fd=descriptor))
            if descriptor is not None:
                os.close(descriptor)
            descriptor = opened
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise PermissionError(f"state directory is not owned by this user: {path}")
            if stat.S_IMODE(info.st_mode) != 0o700:
                os.fchmod(descriptor, 0o700)
        return path
    finally:
        if descriptor is not None:
            os.close(descriptor)
