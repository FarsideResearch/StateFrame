"""No-follow, bounded host read admission for mutation input snapshots."""
from __future__ import annotations

import fnmatch
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

_SECRET = (".env*", "*.pem", "*.key", "id_*", "*.p12", "*credential*",
           "*secret*", ".netrc", ".npmrc", ".pypirc")
_EXCLUDED_DIRS = {".git", ".stateframe"}


def scope_path_matches(path: str, pattern: str) -> bool:
    """Match a grant path; only a complete ** segment may span directories."""
    if not any(char in pattern for char in "*?"):
        return path == pattern

    segments = pattern.split("/")
    states = {0}

    def skip_double_star(indices: set[int]) -> set[int]:
        expanded = set(indices)
        for index in range(len(segments)):
            if index in expanded and segments[index] == "**" and index + 1 < len(segments):
                expanded.add(index + 1)
        return expanded

    states = skip_double_star(states)
    for part in path.split("/"):
        following = set()
        for index in states:
            if index == len(segments):
                continue
            segment = segments[index]
            if segment == "**":
                following.add(index)
                following.add(index + 1)
            elif fnmatch.fnmatchcase(part, segment.replace("[", "[[]")):
                following.add(index + 1)
        states = skip_double_star(following)
        if not states:
            return False
    return len(segments) in states


@dataclass(frozen=True)
class InputSnapshot:
    files: dict[str, Path]
    descriptors: tuple[dict[str, object], ...]
    facts: dict[str, dict[str, bool]]


def _close_preserving_error(fd: int) -> None:
    primary = sys.exception()
    try:
        os.close(fd)
    except OSError as cleanup_error:
        if primary is None:
            raise
        primary.add_note(f"snapshot cleanup close failed: errno={cleanup_error.errno}")


def _parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or len(value) > 4096 or value.startswith("/"):
        raise ValueError("invalid input path")
    parts = tuple(value.split("/"))
    if any(part in ("", ".", "..") or "\\" in part or "\x00" in part for part in parts):
        raise ValueError("invalid input path")
    folded = tuple(part.casefold() for part in parts)
    if any(part in _EXCLUDED_DIRS or any(fnmatch.fnmatchcase(part, pattern) for pattern in _SECRET)
           for part in folded):
        raise ValueError("secret input path is excluded")
    return parts


def _open_at(root_fd: int, parts: tuple[str, ...]) -> int:
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)


def input_id_for(index: int) -> str:
    if type(index) is not int or not 0 <= index < 16:
        raise ValueError("invalid mutation input index")
    return f"in{index}"


def _admitted_parts(path: str, policy: object,
                    ordinary_read_globs: tuple[str, ...] | None) -> tuple[str, ...]:
    parts = _parts(path)
    globs = getattr(policy, "read_globs", ())
    if globs and not any(scope_path_matches(path, pattern) for pattern in globs):
        raise ValueError("input path exceeds mutation read grant")
    if ordinary_read_globs is not None and not any(
            scope_path_matches(path, pattern) for pattern in ordinary_read_globs):
        raise ValueError("input path exceeds ordinary read scope")
    return parts


def input_path_admitted(repo: Path, path: str, policy: object,
                        ordinary_read_globs: tuple[str, ...] | None = None) -> bool:
    """Check proposal eligibility with the same path and file rules as input admission."""
    root_fd = None
    fd = None
    try:
        parts = _admitted_parts(path, policy, ordinary_read_globs)
        root_fd = os.open(Path(repo), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fd = _open_at(root_fd, parts)
        info = os.fstat(fd)
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 4 << 20
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if fd is not None:
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)


def admit_inputs(repo: Path, requests: list[object], policy: object, *, snapshot_root: Path,
                 ordinary_read_globs: tuple[str, ...] | None = None,
                 before_open=None) -> InputSnapshot:
    """Copy admitted repo files into a new private directory before a payload can start."""
    if not isinstance(requests, list) or len(requests) > 16:
        raise ValueError("at most sixteen input files are allowed")
    repo = Path(repo)
    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root = Path(snapshot_root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    files: dict[str, Path] = {}
    descriptors: list[dict[str, object]] = []
    facts: dict[str, dict[str, bool]] = {}
    total = 0
    current_path = None
    snapshot_io_failed = False
    try:
        for index, request in enumerate(requests):
            current_path = request if isinstance(request, str) else (
                request.get("path") if isinstance(request, dict) else None)
            if isinstance(request, str):
                path, tail = request, None
            elif (isinstance(request, dict) and set(request) <= {"path", "tail_bytes"}
                  and "path" in request):
                path, tail = request["path"], request.get("tail_bytes")
            else:
                raise ValueError("invalid input request")
            parts = _admitted_parts(path, policy, ordinary_read_globs)
            if tail is not None and (type(tail) is not int or not 0 < tail <= 4 << 20):
                raise ValueError("invalid tail size")
            if before_open is not None:
                before_open()
            fd = _open_at(root_fd, parts)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4 << 20:
                    raise ValueError("input must be a bounded, singly linked regular file")
                before = (info.st_dev, info.st_ino, info.st_size,
                          info.st_mtime_ns, info.st_ctime_ns)
                offset = max(0, info.st_size - tail) if tail is not None else 0
                os.lseek(fd, offset, os.SEEK_SET)
                data = bytearray()
                while block := os.read(fd, min(1 << 20, (4 << 20) + 1 - len(data))):
                    data.extend(block)
                    if len(data) > 4 << 20:
                        raise ValueError("input changed beyond the size cap")
                after = os.fstat(fd)
                if before != (after.st_dev, after.st_ino, after.st_size,
                              after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("input changed during snapshot")
            finally:
                os.close(fd)
            total += len(data)
            if total > 16 << 20:
                raise ValueError("input set exceeds the size cap")
            ident = input_id_for(index)
            target = root / ident
            out = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
            files[ident] = target
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(out, view):]
                os.fsync(out)
            except OSError:
                snapshot_io_failed = True
                raise
            finally:
                _close_preserving_error(out)
            descriptors.append({"id": ident, "path": "/in/" + ident,
                                "bytes": len(data), "name": parts[-1]})
            wal_present = False
            if parts[-1].lower().endswith((".db", ".sqlite", ".sqlite3")):
                try:
                    wal_fd = _open_at(root_fd, (*parts[:-1], parts[-1] + "-wal"))
                except (OSError, ValueError):
                    pass
                else:
                    try:
                        wal_present = stat.S_ISREG(os.fstat(wal_fd).st_mode)
                    finally:
                        os.close(wal_fd)
            facts[ident] = {"wal_present": wal_present}
        dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        except OSError:
            snapshot_io_failed = True
            raise
        finally:
            _close_preserving_error(dir_fd)
        return InputSnapshot(files, tuple(descriptors), facts)
    except BaseException as error:
        for path in files.values():
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f"snapshot cleanup unlink failed: errno={cleanup_error.errno}")
        try:
            root.rmdir()
        except OSError as cleanup_error:
            error.add_note(f"snapshot cleanup rmdir failed: errno={cleanup_error.errno}")
        if (not snapshot_io_failed and current_path is not None
                and isinstance(error, (OSError, ValueError))):
            raise ValueError(f"{current_path!r}: {error}") from error
        raise
    finally:
        _close_preserving_error(root_fd)
