"""Route historical reads to the immutable snapshot, never redirect writes.

Installed by the guarded audit/test launcher and the bridge. This compatibility
boundary also covers old readers in files explicitly excluded from editing.
Caller-owned temporary results directories and new bridge outputs are unchanged.
"""
from __future__ import annotations

import builtins
import io
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "results"
SNAPSHOT = ROOT / "results.SNAPSHOT-20260920"
_installed = False


def directory_path(descriptor: int) -> str:
    if sys.platform == "darwin":
        import fcntl
        # F_GETPATH is Darwin's stable fd-to-path operation (sys/fcntl.h).
        return os.fsdecode(fcntl.fcntl(descriptor, 50, bytes(1024)).split(b"\0", 1)[0])
    return os.readlink(f"/proc/self/fd/{descriptor}")


def evidence_path(path, *, dir_fd=None):
    if isinstance(path, int) or path is None:
        return path
    raw = os.fsdecode(path)
    if not os.path.isabs(raw) and dir_fd not in (None, -1):
        raw = os.path.join(directory_path(dir_fd), raw)
    absolute = os.path.abspath(raw)
    live = str(LIVE)
    if absolute == live or absolute.startswith(live + os.sep):
        return str(SNAPSHOT) + absolute[len(live):]
    return path


def install_read_routing():
    global _installed
    if _installed:
        return
    _installed = True
    original_builtin, original_io, original_os = builtins.open, io.open, os.open

    def reader(original):
        def opened(file, mode="r", *args, **kwargs):
            if not any(flag in mode for flag in "wax+"):
                file = evidence_path(file)
            return original(file, mode, *args, **kwargs)
        return opened

    def os_open(path, flags, *args, **kwargs):
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            path = evidence_path(path, dir_fd=kwargs.get("dir_fd"))
        return original_os(path, flags, *args, **kwargs)

    builtins.open, io.open, os.open = reader(original_builtin), reader(original_io), os_open
    # Enumeration and existence checks must agree with the bytes readers see.
    for name in ("stat", "lstat", "scandir", "listdir"):
        original = getattr(os, name)
        def routed(path=".", *args, _original=original, **kwargs):
            return _original(evidence_path(path, dir_fd=kwargs.get("dir_fd")), *args, **kwargs)
        setattr(os, name, routed)
