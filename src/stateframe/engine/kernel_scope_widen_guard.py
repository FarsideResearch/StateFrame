"""No-follow file identity for opt-in edits outside the starting scope."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


_ADMITTED: dict[str, tuple[Path, str, tuple[int, int]]] = {}


@contextmanager
def _parent(root: Path, relative: str):
    parts = PurePosixPath(relative).parts
    if (not parts or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise OSError("path is not a normalized repository path")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def _key(path: Path) -> str:
    return str(path.absolute())


def admitted(path: Path) -> bool:
    return _key(path) in _ADMITTED


def forget(path: Path) -> None:
    _ADMITTED.pop(_key(path), None)


def _open(root: Path, relative: str, flags: int, expected: tuple[int, int] | None = None):
    with _parent(root, relative) as (parent, name):
        descriptor = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    info = os.fstat(descriptor)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (expected is not None and (info.st_dev, info.st_ino) != expected)):
        os.close(descriptor)
        raise OSError("widened file identity changed")
    return descriptor, (info.st_dev, info.st_ino)


def admit(root: Path, relative: str) -> tuple[bytes, str]:
    """Read through a verified descriptor, then remember its identity."""
    descriptor, identity = _open(root, relative, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as source:
        data = source.read()
    _ADMITTED[_key(root / relative)] = (root, relative, identity)
    return data, hashlib.sha256(data).hexdigest()


def baseline_digest(root: Path, relative: str) -> str:
    descriptor, _ = _open(root, relative, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def read(path: Path) -> bytes:
    admission = _ADMITTED.get(_key(path))
    if admission is None:
        return path.read_bytes()
    root, relative, expected = admission
    descriptor, _ = _open(root, relative, os.O_RDONLY, expected)
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


def verify(path: Path) -> None:
    admission = _ADMITTED.get(_key(path))
    if admission is None:
        return
    root, relative, expected = admission
    with _parent(root, relative) as (parent, name):
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if ((current.st_dev, current.st_ino) != expected
                or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1):
            raise OSError("widened file identity changed before publication")


def write(path: Path, data: bytes) -> int:
    admission = _ADMITTED.get(_key(path))
    if admission is None:
        return path.write_bytes(data)
    root, relative, expected = admission
    descriptor, _ = _open(root, relative, os.O_WRONLY, expected)
    try:
        verify(path)
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        return len(data)
    finally:
        os.close(descriptor)


def unlink(path: Path) -> None:
    admission = _ADMITTED.get(_key(path))
    if admission is None:
        path.unlink()
        return
    root, relative, expected = admission
    with _parent(root, relative) as (parent, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if ((info.st_dev, info.st_ino) != expected
                    or (current.st_dev, current.st_ino) != expected
                    or info.st_nlink != 1):
                raise OSError("widened file identity changed")
            os.unlink(name, dir_fd=parent)
        finally:
            os.close(descriptor)


def move(source: Path, destination: Path, *, case_only: bool = False) -> None:
    admission = _ADMITTED.get(_key(source))
    if admission is None:
        raise OSError("source has no widening admission")
    root, relative, expected = admission
    destination_relative = destination.relative_to(root).as_posix()
    with _parent(root, relative) as (source_parent, source_name):
        with _parent(root, destination_relative) as (dest_parent, dest_name):
            descriptor = os.open(source_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=source_parent)
            try:
                info = os.fstat(descriptor)
                current = os.stat(source_name, dir_fd=source_parent, follow_symlinks=False)
                if ((info.st_dev, info.st_ino) != expected
                        or (current.st_dev, current.st_ino) != expected
                        or info.st_nlink != 1):
                    raise OSError("widened file identity changed")
                if case_only:
                    destination_info = os.stat(dest_name, dir_fd=dest_parent,
                                               follow_symlinks=False)
                    if (destination_info.st_dev, destination_info.st_ino) != expected:
                        raise OSError("widened move destination identity changed")
                    os.rename(source_name, dest_name, src_dir_fd=source_parent,
                              dst_dir_fd=dest_parent)
                    return
                os.link(source_name, dest_name, src_dir_fd=source_parent,
                        dst_dir_fd=dest_parent, follow_symlinks=False)
                try:
                    current = os.stat(source_name, dir_fd=source_parent,
                                      follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != expected:
                        raise OSError("widened file identity changed")
                    os.unlink(source_name, dir_fd=source_parent)
                except BaseException:
                    os.unlink(dest_name, dir_fd=dest_parent)
                    raise
            finally:
                os.close(descriptor)
