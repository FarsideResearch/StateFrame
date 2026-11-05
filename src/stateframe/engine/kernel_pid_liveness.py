"""Signal-zero liveness that does not count a Linux zombie as alive.

`os.kill(pid, 0)` succeeds for a zombie (State Z): its pid stays reserved until the parent reaps it, but the process
is gone. Temp sweeps that keep a file while its owner pid "lives" would otherwise keep it forever behind an unreaped
child. On Linux the state comes from /proc/<pid>/stat; macOS keeps the plain signal-zero answer.
"""
from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

PROC = Path("/proc")


def linux_zombie(pid: int, proc: Path = PROC) -> bool:
    """True only when /proc shows `pid` in State Z. Unreadable or missing stat means not provably a zombie."""
    try:
        data = (proc / str(pid) / "stat").read_bytes()
    except OSError:
        return False
    # The command name is parenthesised and may itself contain ") "; the state follows the LAST ")".
    _, _, rest = data.rpartition(b")")
    fields = rest.split()
    return bool(fields) and fields[0] == b"Z"


def _linux_group_has_live_member(pgid: int, proc: Path) -> bool:
    """Whether /proc shows a non-zombie member of `pgid`; vanished entries are ignored."""
    for stat_path in proc.glob("*/stat"):
        pid_name = stat_path.parent.name
        if not (pid_name.isascii() and pid_name.isdecimal()):
            continue
        try:
            data = stat_path.read_bytes()
        except OSError:
            continue
        _, separator, rest = data.rpartition(b")")
        if not separator:
            continue
        fields = rest.split()
        if len(fields) < 3:
            continue
        try:
            member_pgid = int(fields[2])
        except ValueError:
            continue
        if member_pgid == pgid and fields[0] != b"Z":
            return True
    return False


def pid_alive(pid: int, *, platform: str = sys.platform, proc: Path = PROC) -> bool:
    """Whether `pid` names a running process. ESRCH is dead; EPERM is alive (someone else's process)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno != errno.EPERM:
            raise
    return not (platform.startswith("linux") and linux_zombie(pid, proc))


def group_alive(pgid: int, *, platform: str = sys.platform, proc: Path = PROC) -> bool:
    """Whether a process group has a live member, excluding Linux zombie-only groups."""
    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH or isinstance(error, ProcessLookupError):
            return False
        if error.errno == errno.EPERM or isinstance(error, PermissionError):
            return True
        raise
    if platform.startswith("linux"):
        return _linux_group_has_live_member(pgid, proc)
    return True


__all__ = ["group_alive", "linux_zombie", "pid_alive"]
