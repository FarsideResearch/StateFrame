"""Identity-checked cleanup of orphan mutation runtime roots."""
from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__:
    from .kernel_exec_tracker import ProcessTable, process_cmdline_digest
    from .kernel_mutate_wire import strict_loads
else:
    from kernel_exec_tracker import ProcessTable, process_cmdline_digest
    from kernel_mutate_wire import strict_loads


@dataclass(frozen=True)
class SweepReport:
    cleaned: tuple[str, ...]
    unresolved: tuple[str, ...]


def _boot_id():
    if sys.platform.startswith("linux"):
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not value:
            raise ValueError("Linux boot identity is unavailable")
        return value
    if sys.platform == "darwin":
        value = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"], timeout=5).decode("ascii").strip()
        if not value:
            raise ValueError("macOS boot identity is unavailable")
        return "macos:" + value
    raise ValueError("host boot identity is unavailable")


def _unresolved(root_fd, reason):
    data = (json.dumps({"kind": "UNRESOLVED", "reason": reason}, sort_keys=True) + "\n").encode()
    fd = os.open("UNRESOLVED.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                 0o600, dir_fd=root_fd)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short unresolved record write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(root_fd)


def _rows(root_fd):
    try:
        fd = os.open("launches.jsonl", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    except FileNotFoundError:
        return []
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 2 << 20:
            raise ValueError("invalid launch ledger file")
        data = bytearray()
        while len(data) < info.st_size:
            block = os.read(fd, info.st_size - len(data))
            if not block:
                raise ValueError("launch ledger changed during read")
            data.extend(block)
    finally:
        os.close(fd)
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete launch ledger")
    rows = []
    for line in data.splitlines():
        value = strict_loads(line)
        if not isinstance(value, dict) or value.get("schema") != "stateframe-mutation-launch/1":
            raise ValueError("invalid launch ledger")
        rows.append(value)
    return rows


def _identities(rows, boot_id, *, require_registered=True):
    by_launch = {}
    history = {}
    for row in rows:
        launch = row.get("launch")
        if not isinstance(launch, str) or not launch:
            raise ValueError("invalid launch id")
        if row.get("state") not in ("PREPARED", "STARTED", "REGISTERED", "RELEASED", "EXITED", "REAPED", "CLEANUP_FAILED"):
            raise ValueError("invalid launch state")
        identity = row.get("identity")
        if identity is not None:
            if not isinstance(identity, dict) or identity.get("boot_id") != boot_id:
                raise ValueError("launch boot identity differs")
        previous_identity = by_launch.get(launch, (None, None))[1]
        by_launch[launch] = (row["state"], identity if identity is not None else previous_identity)
        history.setdefault(launch, set()).add(row["state"])
    if require_registered:
        for launch, (state, identity) in by_launch.items():
            pre_registration_failure = (sys.platform.startswith("linux")
                                        and state == "CLEANUP_FAILED"
                                        and history[launch] <= {"PREPARED", "CLEANUP_FAILED"})
            linux_prepared = sys.platform.startswith("linux") and state == "PREPARED"
            if identity is None and not linux_prepared and not pre_registration_failure:
                raise ValueError("started launch has no recorded birth identity")
    return [identity for _, identity in by_launch.values() if identity is not None]


def _kill_recorded(table, identity):
    pid = identity.get("pid")
    start = identity.get("start")
    if type(pid) is not int or pid <= 0 or not isinstance(start, (list, tuple)) or len(start) != 2:
        raise ValueError("invalid process identity")
    before = table.identity(pid)
    if before is None or before.zombie:
        return
    if list(before.start) != list(start):
        raise ValueError("process identity differs")
    if sys.platform.startswith("linux"):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ValueError("pidfd signaling unavailable")
        namespace = identity.get("ns")
        command_digest = identity.get("cmdline_sha256")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("recorded process PID namespace identity is missing")
        if (not isinstance(command_digest, str) or len(command_digest) != 64
                or any(char not in "0123456789abcdef" for char in command_digest)):
            raise ValueError("recorded process command identity is missing")
        try:
            pidfd = os.pidfd_open(pid)
        except OSError as error:
            raise ValueError("recorded process could not be pinned by pidfd") from error
        try:
            # The pidfd pins the task after open; namespace and launch argv prove it is
            # the recorded sandbox process rather than a same-tick replacement.
            after = table.identity(pid)
            if after is None or after.zombie:
                return
            if after.start != before.start:
                raise ValueError("process identity changed")
            try:
                actual_namespace = table.pid_namespace(pid)
            except OSError as error:
                raise ValueError("process PID namespace identity unavailable") from error
            if actual_namespace != namespace:
                raise ValueError("process PID namespace identity differs")
            try:
                actual_command_digest = process_cmdline_digest(pid)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError("process command identity unavailable") from error
            if actual_command_digest != command_digest:
                raise ValueError("process command identity differs")
            current = table.identity(pid)
            if current is None or current.zombie:
                return
            if current.start != after.start:
                raise ValueError("process identity changed during verification")
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            os.close(pidfd)
    else:
        raise ValueError("pinned process signaling unavailable on macOS")


def _remove_tree(fd, *, parent_fd, name, inode):
    # Admitted bundles are sealed 0500; ownership and inode were checked by caller.
    os.fchmod(fd, 0o700)
    for entry in os.listdir(fd):
        info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                if os.fstat(child).st_ino != info.st_ino or os.fstat(child).st_dev != info.st_dev:
                    raise ValueError("directory inode changed")
                _remove_tree(child, parent_fd=fd, name=entry, inode=(info.st_dev, info.st_ino))
            finally:
                os.close(child)
        elif stat.S_ISREG(info.st_mode):
            current = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("file inode changed")
            os.unlink(entry, dir_fd=fd)
        else:
            raise ValueError("special file or link in runtime root")
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != inode or not stat.S_ISDIR(current.st_mode):
        raise ValueError("runtime root inode changed")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def sweep(rt_parent: Path, *, run_locks: Callable, boot_id: str | None = None,
          dry_run: bool = False) -> SweepReport:
    """Sweep only unlocked roots; unprovable process or inode state leaves a root."""
    parent = Path(rt_parent)
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return SweepReport((), ())
    if dry_run:
        try:
            candidates = tuple(name for name in os.listdir(parent_fd) if not run_locks(name))
            return SweepReport(candidates, ())
        finally:
            os.close(parent_fd)
    cleaned = []
    unresolved = []
    current_boot = boot_id or _boot_id()
    table = ProcessTable()
    try:
        for name in os.listdir(parent_fd):
            try:
                if run_locks(name):
                    continue
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    unresolved.append(name)
                    continue
                root_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    if (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != (info.st_dev, info.st_ino):
                        raise ValueError("runtime root inode changed")
                    owner_fd = os.open("owner.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
                    try:
                        with os.fdopen(owner_fd, "rb") as stream:
                            owner_info = os.fstat(stream.fileno())
                            if not stat.S_ISREG(owner_info.st_mode) or owner_info.st_size > 8192:
                                raise ValueError("invalid owner record")
                            owner = strict_loads(stream.read(8193))
                    except BaseException:
                        raise
                    if not isinstance(owner, dict) or not isinstance(owner.get("boot_id"), str) or not owner["boot_id"]:
                        raise ValueError("invalid owner boot id")
                    ledger = _rows(root_fd)
                    identities = _identities(ledger, owner["boot_id"],
                                             require_registered=owner["boot_id"] == current_boot)
                    owner_identity = owner.get("identity")
                    if owner_identity is None and "pid" in owner:
                        owner_identity = {"pid": owner.get("pid"), "start": owner.get("start"),
                                          "boot_id": owner.get("boot_id"), "ns": owner.get("ns"),
                                          "cmdline_sha256": owner.get("cmdline_sha256")}
                    if owner_identity is not None and (not isinstance(owner_identity, dict)
                                                       or owner_identity.get("boot_id") != owner["boot_id"]):
                        raise ValueError("invalid owner identity")
                    if owner["boot_id"] != current_boot:
                        # No process from a previous kernel boot can still exist.
                        _remove_tree(root_fd, parent_fd=parent_fd, name=name, inode=(info.st_dev, info.st_ino))
                        cleaned.append(name)
                        continue
                    if owner_identity is None:
                        raise ValueError("supervisor birth identity is missing")
                    if owner_identity is not None:
                        identities.append(owner_identity)
                    for identity in identities:
                        _kill_recorded(table, identity)
                    deadline = time.monotonic() + 2
                    while True:
                        live = []
                        for identity in identities:
                            actual = table.identity(identity["pid"])
                            if actual is not None and not actual.zombie and list(actual.start) == list(identity["start"]):
                                live.append(actual.pid)
                        if not live:
                            break
                        if time.monotonic() > deadline:
                            raise ValueError("mutation process survived kill")
                        time.sleep(.02)
                    _remove_tree(root_fd, parent_fd=parent_fd, name=name, inode=(info.st_dev, info.st_ino))
                    cleaned.append(name)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    unresolved.append(name)
                    try:
                        _unresolved(root_fd, str(error))
                    except OSError:
                        pass
                finally:
                    os.close(root_fd)
            except (OSError, ValueError):
                unresolved.append(name)
    finally:
        os.close(parent_fd)
    return SweepReport(tuple(cleaned), tuple(unresolved))
